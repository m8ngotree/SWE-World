import os
import sys
import importlib.util
import time
import gym
import logging
import shlex
from typing import Dict, Any, Tuple, List
from pathlib import Path
import re  # [FIX] Added missing import
import json  # [NEW] For parsing FAIL_TO_PASS/PASS_TO_PASS if stored as JSON string
import random  # [NEW] For random sampling PASS_TO_PASS tests
import concurrent.futures
from tqdm import tqdm
from typing import Tuple, List, Set, Optional

# 为了从patch中解析到修改的文件
from swebench.harness.utils import get_modified_files

# Helper function from the previous response
# \b ensures we match whole words only (e.g., 'python', not 'my-python-script').
PYTHON_CMD_PATTERN = re.compile(r'\b(python\d*|pytest)\b')

def is_python_execution_command(code: str) -> bool:
    """
    判断当前 code 是否是需要收集数据的“python 执行命令”。

    条件：
    1. 必须以 execute_bash 开头
    2. 不能包含 pip install（不区分大小写）
    3. 必须匹配 PYTHON_CMD_PATTERN（包含 python/pytest 等）
    """
    if not code:
        return False

    # SKIP_COMMANDS = ["sed ", "find ", "cp ", "chmod ", "ls ", "git ", "cat ", "mv "]
    SKIP_COMMANDS = ["git "]

    stripped = code.strip()

    # 1. 不能包含 git / sed / find / cp / chmod / ls 等
    if any(skip_cmd in stripped for skip_cmd in SKIP_COMMANDS):
        return False

    # 2. 必须以 execute_bash 开头
    if not stripped.startswith("execute_bash"):
        return False

    # 3. 不能包含 pip install（不区分大小写）
    lowered = stripped.lower()
    if "pip install" in lowered:
        return False

    if "python3 " in stripped:
        return True
    
    if "python " in stripped:
        return True

    # 4. 必须包含 python（或你在 PYTHON_CMD_PATTERN 里定义的命令）
    return bool(PYTHON_CMD_PATTERN.search(stripped))


def _get_files_from_patch(patch_text: str) -> set:
    """Extracts unique file paths from a git diff patch."""
    files = set()
    if not patch_text:
        return files
    # Regex to find file paths in lines like '--- a/path/to/file.py' or '+++ b/path/to/file.py'
    for line in patch_text.split('\n'):
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            # Strip the prefix and add the file path
            path = line.split('\t')[0]  # handle potential extra info after path
            files.add(path.split('/', 1)[1])
    return files


def _load_test_list(raw_value) -> List[str]:
    """
    [NEW] Helper: robustly load FAIL_TO_PASS / PASS_TO_PASS from ds.
    It may be a list or a JSON-encoded string.
    """
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]
    if isinstance(raw_value, str):
        try:
            v = json.loads(raw_value)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            # Fallback: maybe it's a simple string，按单个条目处理
            return [raw_value]
    return []

def _rewrite_testbed_paths(full_command_str: str, repo_path: str) -> str:
    """
    将命令中以 /testbed 开头的路径替换为 repo_path。
    """
    try:
        parts = shlex.split(full_command_str)
    except Exception:
        # fallback：不解析，直接通过字符串替换（可选）
        print(f"[FIX] Failed to parse command string: {full_command_str}")
        return full_command_str.replace(" /testbed", f" {repo_path}")

    if full_command_str.startswith("execute_bash"):
        return full_command_str.replace(" /testbed", f" {repo_path}")

    new_parts = []
    for p in parts:
        if p.startswith("/testbed"):
            rel = p[len("/testbed"):]          # 取得相对路径
            new_path = os.path.join(repo_path, rel.lstrip('/'))
            new_parts.append(new_path)
        else:
            new_parts.append(p)

    # 重建命令字符串
    return " ".join(shlex.quote(x) for x in new_parts)


def _parse_test_script(script_content: str, repo_path: str = "/testbed", is_sim: bool = True) -> Tuple[List[str], str]:
    """
    [NEW] Intelligently parses a run_tests.sh script to extract setup commands
    and the final pytest command, correctly handling here-documents for git apply.
    """
    lines = script_content.split('\n')
    setup_commands = [f"cd {repo_path}"]
    pytest_command = None
    
    in_here_document = False
    here_doc_marker = None
    current_command_block = []

    for line in lines:
        stripped_line = line.strip()

        if "pytest" in stripped_line: # 如果包含pytest，说明前期准备完成，后面的全部可以跳过
            pytest_command = stripped_line
            break

        
        if " /testbed" in stripped_line and is_sim:
            stripped_line = _rewrite_testbed_paths(stripped_line, repo_path)

        if in_here_document:
            current_command_block.append(line)
            if stripped_line == here_doc_marker:
                setup_commands.append("\n".join(current_command_block))
                current_command_block = []
                in_here_document = False
                here_doc_marker = None
            continue

        if not stripped_line or stripped_line.startswith(('#', ':', 'source', 'conda', 'pip')):
            continue

        if PYTHON_CMD_PATTERN.search(stripped_line):
            pytest_command = stripped_line # Assume the last python/pytest command is the one we want
            continue

        if stripped_line.startswith(('git', 'cd')):
            # Check for here-document start
            if '<<' in stripped_line:
                try:
                    marker = stripped_line.split('<<')[1].strip().replace("'", "")
                    in_here_document = True
                    here_doc_marker = marker
                    current_command_block.append(stripped_line)
                except IndexError:
                    setup_commands.append(stripped_line) # Malformed here-doc, treat as normal command
            else:
                setup_commands.append(stripped_line)
    
    return setup_commands, pytest_command



def extract_init_and_test_swebv(eval_commands: List[str], test_command_index = -3, init_command_end_index = -4, repo_path: str = "/testbed", is_sim: bool = False) -> Tuple[List[str], str]:
    """
    从 make_eval_script_list_py 生成的 eval_commands 中提取：
    1. 初始化设置（仅包含文件/仓库相关操作，比如 cd、git），并且包含：
        - reset_tests_command（第一次出现）
        - apply_test_patch_command
    2. test_command（真正执行测试的命令）

    假设 eval_commands 的结构是：
        [... 前面是各种初始化命令 ...,
         reset_tests_command,          # n-6
         apply_test_patch_command,     # n-5
         ": '<START_TEST_OUTPUT>'",    # n-4
         test_command,                 # n-3
         ": '<END_TEST_OUTPUT>'",      # n-2
         reset_tests_command]          # n-1

    因此：
        - test_command 在倒数第 3 个位置（索引 -3）
        - 倒数后 4 个是 [": START", test_command, ": END", reset_reset]
        - 我们希望 init 部分 = eval_commands[:-4]
    """

    if len(eval_commands) < 6:
        raise ValueError("eval_commands 结构不符合预期，长度不足 6。")

    # 1. 提取 test_command：根据当前生成逻辑，它就是倒数第 3 个
    # test_command = eval_commands[-3]
    test_command = eval_commands[test_command_index]

    # 2. 提取原始初始化命令：包含前面的初始化 + 第一次 reset + apply
    # raw_init_commands = eval_commands[:-4]
    raw_init_commands = eval_commands[:init_command_end_index]

    # 需要过滤掉的“环境初始化”命令前缀
    ENV_PREFIXES = (
        "source ",   # source /opt/miniconda3/bin/activate
        "conda ",    # conda activate xxx
        "python ",   # python xxx.py
        "pip ",      # pip install xxx
    )

    # 认为是“文件/仓库操作”的命令前缀（可以按需扩展）
    FILE_PREFIXES = (
        "cd ",
        "git ",
        "ls ",
        "rm ",
        "mv ",
        "cp ",
    )

    init_commands: List[str] = []
    for cmd in raw_init_commands:
        stripped = cmd.lstrip()

        # 过滤掉环境相关命令
        if stripped.startswith(ENV_PREFIXES):
            continue

        # 只保留文件/仓库相关命令（比如 cd、git 等）
        if stripped.startswith(FILE_PREFIXES):
            if " /testbed" in stripped and is_sim: # 替换路径，因为模拟的时候路径不是/testbed
                stripped = _rewrite_testbed_paths(stripped, repo_path)

            init_commands.append(stripped)

    return init_commands, test_command



def get_test_files_from_spec(
    f2p_raw: str,
    p2p_raw: str,
    test_command: str,
    test_patch: str
) -> list:
    """
    根据 f2p, p2p, test_command 和 test_patch 提取要运行的测试文件路径列表。
    
    Args:
        f2p_raw: FAIL_TO_PASS 的原始 JSON 字符串
        p2p_raw: PASS_TO_PASS 的原始 JSON 字符串
        test_command: 运行测试的 shell 命令字符串
        test_patch: 测试 patch 字符串
        
    Returns:
        list: 包含测试文件路径的列表 (例如 ['tests/test_models.py'])
    """

    INTERESTING_EXTS = [".py", ".rst", ".txt"]

    # 内部辅助函数：安全解析 JSON 列表
    def safe_load_json(json_str):
        if not json_str:
            return []
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse JSON: {json_str}")
            return json_str

    f2p_list = safe_load_json(f2p_raw)
    p2p_list = safe_load_json(p2p_raw)
    labels = [str(x) for x in (f2p_list + p2p_list)]
    
    found_files: set[str] = set()

    # ==============================================================
    # 从 F2P 和 P2P 中提取显式的文件路径
    # ==============================================================
    # 适用格式: "tests/test_pycode_ast.py::test_unparse" 或 "path/to/test.py"
    # 正则逻辑: 匹配以 任意扩展名 结尾的路径，后面可能跟 ::
    # 例如: tests/foo.py::..., tests/roots/bar.rst::...
    # ==============================================================
    path_pattern = re.compile(r'^([\w\-\./]+\.[A-Za-z0-9]+)(?:::)?')
    
    for item in labels:
        match = path_pattern.match(item)
        if match:
            found_files.add(match.group(1))
            
    # print("==============")
    # print(f"found_files: {found_files}")

    # ==============================================================
    # 从 test_patch 中提取修改的文件
    # ==============================================================
    modified_files = get_modified_files(test_patch)
    found_files.update(modified_files)
    # print(f"found_files: {found_files}")

    # ==============================================================
    # 准备工作: 解析 test_command
    # ==============================================================
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        tokens = test_command.split()

    # ==============================================================
    # 所有场景：从 Test Command 中提取所有“看起来是文件”的 token
    # （包括 .py / .rst / .txt 等）
    # ==============================================================
    cmd_explicit_files: list[str] = []
    
    for token in tokens:
        # 1) 跳过选项参数，例如 -v, --maxfail=1 等
        if token.startswith('-'):
            continue

        # 2) 跳过类似 "PYTHONWARNINGS=xxx" 这种纯环境变量（没有路径）
        if '=' in token and '/' not in token:
            continue
        
        # 3) 看看这个 token 的“最后一段”是不是带扩展名的文件名
        basename = token.split('/')[-1]
        _, ext = os.path.splitext(basename)

        if ext in INTERESTING_EXTS:
            cmd_explicit_files.append(token)
    
    found_files.update(cmd_explicit_files)
    # print(f"found_files: {found_files}")

    # ==============================================================
    # 最后一层过滤：只保留“真正像文件路径”的项
    # ==============================================================
    cleaned_files: set[str] = set()
    for path in found_files:
        basename = path.split('/')[-1]
        _, ext = os.path.splitext(basename)
        if ext in INTERESTING_EXTS:  # 有扩展名才认为是文件
            path = path.lstrip("./")
            cleaned_files.add(path)

    # print(f"cleaned_files: {cleaned_files}")

    # ==============================================================
    # 额外检查（主要用于 Django 场景）：
    #   检查 test_command 和 f2p/p2p 中出现的“模块名”是否能映射到这些文件上
    #   如果不能，则把模块名转成对应的 test file 并加入 cleaned_files
    # ==============================================================

    # 是否是 Django 的 runtests 场景
    is_django_runtests = any(t.endswith("runtests.py") for t in tokens) or "runtests.py" in test_command

    if is_django_runtests:
        expected_modules: set[str] = set()

        # 1) 从 test_command 中提取模块名（不含 /，不含 =，包含 .，且不是选项）
        for token in tokens:
            if token.startswith('-'):
                continue
            if '/' in token:
                continue
            if '=' in token:
                continue
            if '.' in token:
                expected_modules.add(token)

        # 2) 从 F2P/P2P 中提取 unittest 风格的 module 部分：
        unittest_pattern = re.compile(
            r"^(?P<test_name>.+?) \((?P<module>[\w\.]+)\.(?P<class_name>[\w_]+)\)$"
        )
        for label in labels:
            m = unittest_pattern.match(label.strip())
            if m:
                expected_modules.add(m.group("module"))

        # 3) 根据 cleaned_files 推导出“文件对应的模块前缀”
        modules_from_files: set[str] = set()
        for path in cleaned_files:
            rel = path.lstrip("./")
            if rel.startswith("tests/"):
                rel = rel[len("tests/") :]
            if rel.endswith(".py"):
                rel = rel[:-3]
            modules_from_files.add(rel.replace("/", "."))

        # 小工具：把 module 名转为一个或多个 test file 路径
        def module_to_test_files(mod: str) -> list[str]:
            parts = mod.split(".")
            if not parts:
                return []
            files: list[str] = []

            # 完整路径：tests/a/b/c/.../last.py
            full_rel = "/".join(parts)
            
            # 按照标准的test模块来说，应当是大于等于三层才转为文件，其余的转为路径 mark一下
            files.append(f"tests/{full_rel}.py")
            
            # # 额外截断为“最多三层”：tests/part0/part1.py
            # if len(parts) > 2:
            #     short_rel = "/".join(parts[:2])
            #     files.append(f"tests/{short_rel}.py")

            return files

        # 4) 检查模块是否能匹配到已有文件，如果不能则生成对应的 test file
        first_ok_check = True
        for mod in sorted(expected_modules):
            ok = any(
                mod == fm
                or mod.startswith(fm + ".")
                or fm.startswith(mod + ".")
                for fm in modules_from_files
            )
            
            if not ok:
                first_ok_check = False
                # print(
                #     f"[WARN] 模块名 '{mod}' 无法在推导出的测试文件模块中匹配：{sorted(modules_from_files)}"
                # )

                # 把这个 module 转成对应的 test file 路径
                new_paths = module_to_test_files(mod)
                for np in new_paths:
                    # 只收 .py
                    base = np.split("/")[-1]
                    _, ext = os.path.splitext(base)
                    if ext in INTERESTING_EXTS:
                        cleaned_files.add(np)

                        # 更新 modules_from_files，避免后续模块继续误报
                        rel = np.lstrip("./")
                        if rel.startswith("tests/"):
                            rel2 = rel[len("tests/") :]
                        else:
                            rel2 = rel
                        if rel2.endswith(".py"):
                            rel2 = rel2[:-3]
                        modules_from_files.add(rel2.replace("/", "."))

        # print(f"first_ok_check: {first_ok_check}")
        if not first_ok_check:
            
            second_ok_check = True
            for mod in sorted(expected_modules):
                ok = any(
                    mod == fm
                    or mod.startswith(fm + ".")
                    or fm.startswith(mod + ".")
                    for fm in modules_from_files
                )
                if not ok:
                    second_ok_check = False
                    break
            # print(f"second_ok_check: {second_ok_check}")
            

    # print(f"cleaned_files: {cleaned_files}")
    return sorted(cleaned_files)




def resolve_module_path(name: str, base_path: str):
    """
    根据模块名和基准路径，判断它对应的是 .py 文件还是包目录，
    并返回 (kind, path)

    :param name: 模块名，比如 "tests.test_math"、"mypkg.helpers.string_helper"
    :param base_path: 作为项目根路径，比如 "/path/to/project"
    :return: (kind, resolved_path)
             kind 为 "file" 或 "dir"
             resolved_path 为 **相对于 base_path 的路径**
    """
    # 规范化基准路径
    base_path = os.path.abspath(base_path)

    # 临时把 base_path 插到 sys.path 最前面，这样 importlib 能在这个目录下找模块
    inserted = False
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
        sys.path.insert(0, os.path.join(base_path, "tests"))
        #print(f"sys.path: {sys.path}")
        inserted = True

    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ImportError(f"Module {name!r} not found under {base_path!r}")

        # 包（目录）：有 submodule_search_locations
        if spec.submodule_search_locations is not None:
            # 包目录通常就是这个列表里的第一个路径
            package_dir_abs = os.path.abspath(list(spec.submodule_search_locations)[0])
            # 转成相对于 base_path 的路径
            package_dir_rel = os.path.relpath(package_dir_abs, base_path)
            return "dir", package_dir_rel

        # 普通模块：origin 是 .py 文件
        origin = spec.origin
        if origin is None:
            # 有些内置模块可能没有 origin，这里简单处理一下
            raise ImportError(f"Module {name!r} has no origin (maybe builtin)")
        origin_abs = os.path.abspath(origin)
        # 转成相对于 base_path 的路径
        origin_rel = os.path.relpath(origin_abs, base_path)
        return "file", origin_rel

    finally:
        # 用完把我们加进去的 base_path 移除，避免影响外部环境
        if inserted and base_path in sys.path:
            sys.path.remove(base_path)
        if inserted and os.path.join(base_path, "tests") in sys.path:
            sys.path.remove(os.path.join(base_path, "tests"))



def prefix_before_first_upper(name: str) -> str:
    """
    从左到右寻找第一个包含大写字母的片段，
    返回它前面的部分（用 . 拼接）。

    约定：
    - 如果没有任何片段包含大写字母：返回原字符串 name
    - 如果第一个片段本身就包含大写字母：返回空字符串 ""
    """
    parts = name.split('.')

    for i, part in enumerate(parts):
        # 只要这个片段里有任意一个字符是大写，就认为它是“类/属性部分”的起点
        if any(ch.isupper() for ch in part):
            # parts[:i] 可能为空列表，这时 join 结果就是 ""
            return '.'.join(parts[:i])

    # 没有任何大写，认为整串都是“模块名”
    return name



def extract_exec_targets_from_code(
    code: str,
    repo_path: str,
    execution_mode: str = "simulated",
    runtime: Optional[object] = None,
) -> Tuple[List[str], List[str], bool]:
    """
    从一条 shell 命令字符串中，解析出：
      1. 参与执行的文件路径列表（含 .py / .rst / .txt）
      2. 参与执行的目录路径列表
      3. 是否存在 python -c 或 << EOF 这种“内联代码执行”的标志位

    Args:
        code: 待解析的命令
        repo_path: 本地仓库路径（用于 simulated 模式下的检查）
        execution_mode: 执行模式，如果不是 "simulated"，则尝试使用 runtime 检查路径
        runtime: DockerRuntime 或类似实例，需实现 path_exists(path, path_type) 方法
    """

    INTERESTING_FILE_EXTENSIONS = {".py", ".rst", ".txt", ".dat", ".html"}
    # 增加一些如果是命令起始符就应该忽略的 token
    IGNORED_COMMANDS = {
        "sudo",
        "bash",
        "sh",
        "execute_bash",
        "timeout",
        "xargs",
        "grep",
        "find",
        "cat",
        "echo",
        "ls",
        "cp",
        "mv",
        "sed",
        "chmod",
        "git",
    }

    file_paths: Set[str] = set()
    dir_paths: Set[str] = set()
    file_path_additions: Set[str] = set()
    dir_path_additions: Set[str] = set()
    module_candidates: Set[str] = set()  # 收集所有“模块名字”候选
    has_inline_code = False

    # 1. 外层 split
    try:
        cmd_parts = shlex.split(code)
    except ValueError:
        cmd_parts = code.split()

    # 2. 遍历每一段
    for part in cmd_parts:
        try:
            inner_cmd = shlex.split(part)
        except ValueError:
            # 如果 shlex 解析失败（例如 heredoc 中引号不匹配），回退到空格分割
            # 但为了避免 split 切碎 python 代码中的字符串，这里做得保守一点
            inner_cmd = part.split()

        i = 0
        while i < len(inner_cmd):
            inner_part = inner_cmd[i]

            # ---------- 0) 预处理 / 垃圾过滤 ----------

            # 遇到 Heredoc 标记 <<，说明后面是内联内容
            if inner_part == "<<":
                has_inline_code = True
                # Heredoc 通常结构是: << 'EOF' \n content \n EOF
                # 很难精确跳过 content，但我们可以标记 inline_code，并尽量跳过紧随其后的 delimiter
                if i + 1 < len(inner_cmd):
                    i += 2
                else:
                    i += 1
                continue

            # 遇到 Pipe 或常见操作符，跳过
            if inner_part in ("|", "&&", "||", ";", "&"):
                i += 1
                continue

            # 遇到重定向符号 (包括 2>/dev/null 这种粘连在一起的)
            # 只要包含 > 或 < 且不是箭头(->, 这种情况在文件名少见，在代码多见)，或者是 /dev/null
            if ">" in inner_part or "<" in inner_part or "/dev/null" in inner_part:
                i += 1
                continue

            # 遇到代码特征符号 (括号、引号、等号)，极大概率是内联代码而不是路径
            # 例如: print(fFAILED...) 或 "filterwarnings"
            if any(c in inner_part for c in "(){}'\""):
                i += 1
                continue

            # 遇到 Shebang 或注释
            if inner_part.startswith("#"):
                i += 1
                continue

            # ---------- 1) 上下文感知跳过 ----------

            # 跳过 cd 目标
            if inner_part == "cd":
                if i + 1 < len(inner_cmd):
                    i += 2
                    continue
                else:
                    i += 1
                    continue

            # 检测 python -c
            if inner_part in ("python", "python3") and i + 1 < len(inner_cmd) and inner_cmd[i + 1] == "-c":
                has_inline_code = True
                i += 3
                continue

            # 清理 candidate (去除末尾的分号、逗号等，解决 reproduce_issue.py; 问题)
            candidate = inner_part.rstrip(";,")

            # 去掉 pytest 的 node id
            if "::" in candidate:
                candidate = candidate.split("::", 1)[0]

            # ---------- 2) 基础过滤 ----------

            # 过滤通配符 (解决 *.py 问题)
            if "*" in candidate:
                i += 1
                continue

            # 过滤参数标志 (e.g., -v, --help, -name)
            if candidate.startswith("-"):
                i += 1
                continue

            # 过滤常见命令名
            if candidate in IGNORED_COMMANDS:
                i += 1
                continue

            # 过滤环境变量赋值 (VAR=VAL)
            if "=" in candidate:
                if "/" not in candidate:
                    i += 1
                    continue
                candidate = candidate.split("=", 1)[1]

            basename = os.path.basename(candidate)
            _, ext = os.path.splitext(basename)

            # ---------- 3) 特判 runtests.py ----------
            # 需要在“显式文件”之前处理，否则 .py 分支会先 continue 掉
            if basename == "runtests.py":
                # 把 runtests.py 本身当作文件
                normalized = candidate.lstrip("./") if candidate.startswith("./") else candidate
                file_paths.add(normalized)

                # 如果后面跟着模块 / 文件参数
                if i + 1 < len(inner_cmd):
                    module_or_file = inner_cmd[i + 1].rstrip(";,")
                    # 如果参数本身就是文件（如 runtests.py some_file.py），直接视为文件
                    if os.path.splitext(module_or_file)[1] in INTERESTING_FILE_EXTENSIONS:
                        normalized_arg = (
                            module_or_file.lstrip("./") if module_or_file.startswith("./") else module_or_file
                        )
                        file_paths.add(normalized_arg)
                    else:
                        # 否则当作模块名，晚点统一处理
                        module_candidates.add(module_or_file)

                i += 2
                continue

            # ---------- 优先判断：显式文件 ----------
            # 如果明确以 .py 等结尾，直接视为文件，不进行后续的模块解析
            # 这解决了 manage.py 被解析为 tests/manage/py.py 的问题
            if ext in INTERESTING_FILE_EXTENSIONS:
                normalized = candidate.lstrip("./") if candidate.startswith("./") else candidate
                file_paths.add(normalized)
                i += 1
                continue

            # ---------- 情况 B：目录 ----------
            # 目录判断增强
            has_slash = "/" in candidate
            is_current_dir = candidate == "."

            if (has_slash or is_current_dir) and "." not in basename and not candidate.startswith("/"):
                normalized = candidate.rstrip("/")
                if normalized.startswith("./"):
                    normalized = normalized[2:]

                if normalized:
                    dir_paths.add(normalized)

                # 如果是目录，就不需要继续作为模块解析了
                i += 1
                continue

            # print(f"file_path: {file_paths}, dir_paths: {dir_paths}")

            # ---------- 4) 识别一般形式的测试模块参数 ----------
            # 例如:
            #   python -m django test tests.delete.tests
            #   python -m django test tests.user_commands.tests.CommandTests.test_call_command_option_parsing
            #   python -m unittest tests.schema.tests.SchemaTests.test_unique_together
            if "." in candidate and "/" not in candidate and not " " in candidate and not "*" in candidate and not ":" in candidate and not "=" in candidate and not "," in candidate:
                # 只记录模块名，真正的路径推断放到后处理阶段
                module_candidates.add(candidate)
                i += 1
                continue

            i += 1

    # ---------- 5) 基于收集到的模块名，生成更多候选路径 ----------

    def add_module_path_variants(path_fragment: str):
        """
        （现在不直接往全局里加，而是用于构造每个 module 的候选列表）
        """
        path_fragment = path_fragment.strip("/")
        if not path_fragment:
            return None, None

        # 统一去掉开头的 tests/，避免生成 tests/tests/...
        short = path_fragment
        if short.startswith("tests/"):
            short = short[len("tests/") :]

        # 认为测试代码都在 tests/ 下
        file_path = f"tests/{short}.py"
        dir_path = f"tests/{short}"
        return file_path, dir_path

    # 小工具：按 execution_mode / runtime 检查路径是否存在
    def _path_exists(path: str, path_type: str) -> bool:
        if execution_mode != "simulated" and runtime is not None:
            try:
                return bool(runtime.path_exists(path, path_type=path_type))
            except Exception:
                return False
        full_path = os.path.join(repo_path, path)
        if path_type == "file":
            return os.path.isfile(full_path)
        return os.path.isdir(full_path)

    # print("module_candidates:", module_candidates)
    for module in module_candidates:
        raw_mod = module.strip()
        if not raw_mod:
            continue

        if raw_mod.startswith("tests."):
            raw_mod = raw_mod[len("tests.") :]

        # 1) 模块名变体集合
        variants: Set[str] = set()
        variants.add(raw_mod) # 原始的模块

        # 2) 大写截断版本 (针对 ClassName.method)
        try:
            base_for_upper = raw_mod
            mod_no_upper = prefix_before_first_upper(base_for_upper)
            if mod_no_upper and mod_no_upper != base_for_upper:
                variants.add(mod_no_upper)
        except NameError:
            # 没有 prefix_before_first_upper 也无所谓
            pass
        
        for mod_variant in variants:
            base = mod_variant.replace(".", "/").rstrip("/")
            if not base:
                continue
            file_path, dir_path = add_module_path_variants(base)
            if file_path:
                file_paths.add(file_path)
            if dir_path:
                dir_paths.add(dir_path)

        # 最终决定注释掉这段代码，因为会出现：resolve module: tests.migrations under path: /tmp/tmp_f_f_f21/testbed, kind: dir, resolved_path: ../../tmp2x07jeps/testbed/tests/migrations
        # 这种是因为sys.path互相冲突了
        # 3) resolve_module_path 作为补充（用 import 真实模块来反推路径）
        #    只在 simulated 模式下使用；非 simulated 模式下以 runtime 文件系统为准
        # if execution_mode == "simulated":
        #     try:
        #         modules_to_resolve = {f"tests.{raw_mod}"}
        #         for mod_name in modules_to_resolve:
        #             try:
        #                 kind, resolved_path = resolve_module_path(mod_name, repo_path)
        #                 print(
        #                     f"resolve module: {mod_name} under path: {repo_path}, "
        #                     f"kind: {kind}, resolved_path: {resolved_path}"
        #                 )
        #                 if kind == "file":
        #                     file_paths.add(resolved_path)
        #                 elif kind == "dir":
        #                     dir_paths.add(resolved_path)
        #             except Exception as e:
        #                 print(f"resolve module: {mod_name} under path: {repo_path}, error: {e}")
        #                 continue
        #     except NameError:
        #         # 没有 resolve_module_path 就跳过这一步
        #         pass

    # 3) 每个模块变成路径，并右向左逐步回退：
    # a.b.c  ->  a/b/c, a/b, a
    for module in module_candidates:
        base = module.replace(".", "/").rstrip("/")
        if not base:
            continue
        if base.startswith("tests/"):
            base = base[len("tests/"):]
        parts = base.split("/")
        # length 从大到小，保证右向左收缩，最多回退两次
        min_length = max(len(parts) - 2, 1)
        for length in range(len(parts), min_length-1, -1):
            sub_path = "/".join(parts[:length])
            file_path, dir_path = add_module_path_variants(sub_path)
            # print(f"回退： file_path: {file_path}, dir_path: {dir_path}")
            if file_path and file_path not in file_path_additions:
                if _path_exists(file_path, "file"):
                    file_path_additions.add(file_path)
                    break
            if dir_path and dir_path not in dir_path_additions:
                if _path_exists(dir_path, "dir"):
                    dir_path_additions.add(dir_path)
                    break
    
    # print(f"file_path_additions: {file_path_additions}")
    # print(f"dir_path_additions: {dir_path_additions}")

    # 把每个 module 推断出的「最佳路径」合并进主集合
    file_paths.update(file_path_additions)
    dir_paths.update(dir_path_additions)

    # ---------- 6) 最终存在性检查 ----------
    # 根据 execution_mode / runtime 选择检查方式，只保留真实存在的路径
    final_files: List[str] = []
    final_dirs: List[str] = []

    if execution_mode != "simulated" and runtime is not None:
        # 使用容器/真实运行环境的 path_exists
        for f in file_paths:
            try:
                if runtime.path_exists(f, path_type="file"):
                    final_files.append(f)
            except Exception:
                # 出错就保守跳过
                continue
        for d in dir_paths:
            try:
                if runtime.path_exists(d, path_type="dir"):
                    final_dirs.append(d)
            except Exception:
                continue
    else:
        # 本地模拟：用 repo_path + os.path 检查
        for f in file_paths:
            if os.path.isfile(os.path.join(repo_path, f)):
                final_files.append(f)
        for d in dir_paths:
            if os.path.isdir(os.path.join(repo_path, d)):
                final_dirs.append(d)

    # for debug
    # final_files = file_paths
    # final_dirs = dir_paths

    return sorted(final_files), sorted(final_dirs), has_inline_code



def extract_commands_from_log(file_path):
    """
    读取日志文件，移除每行末尾的 sim_env.py:XXX 标记，
    返回清理后的命令列表。
    """
    commands = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 1. 去除行首尾的基本空白符（换行符等）
                line = line.strip()
                
                # 2. 如果行是空的，跳过
                if not line:
                    continue
                
                # 3. 查找分割关键字
                # 这里假设文件名始终是 sim_env.py，如果有变动可以使用正则表达式
                separator = "sim_env.py:"
                
                if separator in line:
                    # 4. 分割字符串，取前半部分（索引0）
                    # 例如: "cmd...   sim_env.py:946" -> ["cmd...   ", "946"]
                    content = line.split(separator)[0]
                    
                    # 5. 再次去除分割后留下的尾部空格
                    clean_content = content.strip()
                    
                    if clean_content:
                        commands.append(clean_content)
                else:
                    # 如果某行没有这个标记，根据需求决定是否保留
                    # 这里默认保留原行内容（如果不是纯空行）
                    commands.append(line)
                    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return []

    return commands

if __name__ == "__main__":

    # command = "execute_bash --cmd 'cd /testbed && for i in 1 2 3; do echo \"=== Run $i ===\"; python reproduce_issue.py; echo; done'"
    # # # print(_rewrite_testbed_paths(command, "/tmp/tmp1231/testbed"))

    command = "execute_bash 'cd /testbed && python runtests.py tests.db_functions.datetime.test_extract_trunc.DateFunctionWithTimeZoneTests -v 2'   "
    print(extract_exec_targets_from_code(command, "/tmp/tmp1231/testbed/"))
    # #command = "execute_bash --cmd 'cd /testbed && python -m django test tests.invalid_models_tests.test_models --settings=tests.test_sqlite' "
    # print(prefix_before_first_upper("tests.test_math.TestAdd.test_xxx"))
    # # -> "tests.test_math"

    # print(prefix_before_first_upper("tests.test_math.testAdd.test_xxx"))

    # print(prefix_before_first_upper("pkg.subpkg.module.ClassName.mEthod"))
    # # -> "pkg.subpkg.module"

    # print(prefix_before_first_upper("mypkg.module"))   # 没有大写
    # # -> "mypkg.module"

    # print(prefix_before_first_upper("ClassName.method"))
    # # -> ""
    
    # print(extract_exec_targets_from_code(command, "/tmp/tmp1231/testbed/"))

    # commands = extract_commands_from_log(file_path)

    # for cmd in commands:
    #     exec_files, exec_dirs, hasinline_code = extract_exec_targets_from_code(cmd, "/tmp/django", "simulated")
    #     print(f"raw command: {cmd}, exec_files: {exec_files}, exec_dirs: {exec_dirs}, hasinline_code: {hasinline_code}")