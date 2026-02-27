import re
from typing import Dict
from r2egym.agenthub.action import Action
from r2egym.agenthub import CONTINUE_MSG


class Observation:
    def __init__(
        self,
        bash_output,
        error_code,
        action: Action,
        num_lines: int = 40,
        raw_simulation: str = None,   # 新增：可选的 raw_simulation
    ):
        self.bash_output = bash_output
        self.error_code = error_code
        self.action = action
        self.num_lines = num_lines
        self.raw_simulation = raw_simulation  # 新增：保存原始 LLM 响应

    def __str__(self):
        # empty or no function call
        if not self.action.function_name:
            return CONTINUE_MSG
        elif self.action.function_name == "finish" or self.action.function_name == "submit":
            return "<<< Finished >>>"
        else:
            if self.action.function_name == "execute_bash" or self.action.function_name == "bash":
                lines = self.bash_output.splitlines() if self.bash_output else []
                if len(lines) > 2 * self.num_lines:
                    top_lines = "\n".join(lines[:self.num_lines])
                    bottom_lines = "\n".join(lines[-self.num_lines:])
                    divider = "-" * 50
                    truncated_output = (
                        f"{top_lines}\n"
                        f"{divider}\n"
                        f"<Observation truncated in middle for saving context>\n"
                        f"{divider}\n"
                        f"{bottom_lines}"
                    )
                else:
                    truncated_output = self.bash_output
                output = (
                    f"Exit code: {self.error_code}\n"
                    f"Execution output of [{self.action.function_name}]:\n"
                    f"{truncated_output}"
                )
            else:
                output = f"Execution output of [{self.action.function_name}]:\n{self.bash_output}"

            # 新增：如果有 raw_simulation，就追加到最后
            # if self.raw_simulation:
            #     output = (
            #         f"{output}\n\n"
            #         f"--- RAW SIMULATION ---\n"
            #         f"{self.raw_simulation}"
            #     )

            return output
