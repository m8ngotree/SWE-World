"""
constants and config for different agents
"""

# supported repos
SUPPORTED_REPOS = [
    # ── Python (original) ──────────────────────────────────────────────────
    "sympy",
    "pandas",
    "pillow",
    "numpy",
    "tornado",
    "coveragepy",
    "aiohttp",
    "pyramid",
    "datalad",
    "scrapy",
    "orange3",
    # ── JavaScript (Multi-SWE-bench JS subset) ─────────────────────────────
    "expressjs/express",
    "lodash/lodash",
    "moment/moment",
    "axios/axios",
    "chalk/chalk",
    "mochajs/mocha",
    "koajs/koa",
    "hapijs/hapi",
    "fastify/fastify",
    "socketio/socket.io",
    "nodemon/nodemon",
    "browserify/browserify",
    # ── TypeScript (Multi-SWE-bench TS subset) ─────────────────────────────
    "microsoft/TypeScript",
    "eslint/eslint",
    "prettier/prettier",
    "webpack/webpack",
    "jestjs/jest",
    "nestjs/nest",
    "typeorm/typeorm",
    "inversify/InversifyJS",
    "ReactiveX/rxjs",
    # ── Rust (Multi-SWE-bench Rust subset) ─────────────────────────────────
    "tokio-rs/tokio",
    "serde-rs/serde",
    "actix/actix-web",
    "hyperium/hyper",
    "diesel-rs/diesel",
    "reqwest-rs/reqwest",
    # ── Go (Multi-SWE-bench Go subset) ─────────────────────────────────────
    "gin-gonic/gin",
    "go-gorm/gorm",
    "spf13/cobra",
    "gorilla/mux",
    "gohugoio/hugo",
    # ── Java (Multi-SWE-bench Java subset) ─────────────────────────────────
    "spring-projects/spring-framework",
    "junit-team/junit5",
    "mockito/mockito",
    "FasterXML/jackson-databind",
    "google/guava",
]

# Mapping from repo full name / short name to language string.
# Used by detect_language() as a fast lookup before falling back to
# command-pattern matching.
LANGUAGE_MAP: dict = {
    # Python
    "sympy": "python", "pandas": "python", "pillow": "python",
    "numpy": "python", "tornado": "python", "coveragepy": "python",
    "aiohttp": "python", "pyramid": "python", "datalad": "python",
    "scrapy": "python", "orange3": "python",
    # JavaScript
    "expressjs/express": "javascript", "express": "javascript",
    "lodash/lodash": "javascript", "lodash": "javascript",
    "moment/moment": "javascript", "moment": "javascript",
    "axios/axios": "javascript", "axios": "javascript",
    "chalk/chalk": "javascript", "chalk": "javascript",
    "mochajs/mocha": "javascript", "mocha": "javascript",
    "koajs/koa": "javascript", "koa": "javascript",
    "hapijs/hapi": "javascript", "hapi": "javascript",
    "fastify/fastify": "javascript", "fastify": "javascript",
    "socketio/socket.io": "javascript",
    "nodemon/nodemon": "javascript",
    "browserify/browserify": "javascript",
    # TypeScript
    "microsoft/typescript": "typescript", "typescript": "typescript",
    "eslint/eslint": "typescript", "eslint": "typescript",
    "prettier/prettier": "typescript", "prettier": "typescript",
    "webpack/webpack": "typescript", "webpack": "typescript",
    "jestjs/jest": "typescript", "jest": "typescript",
    "nestjs/nest": "typescript", "nestjs": "typescript",
    "typeorm/typeorm": "typescript", "typeorm": "typescript",
    "inversify/inversifyjs": "typescript",
    "reactivex/rxjs": "typescript", "rxjs": "typescript",
    # Rust
    "tokio-rs/tokio": "rust", "tokio": "rust",
    "serde-rs/serde": "rust", "serde": "rust",
    "actix/actix-web": "rust", "actix-web": "rust",
    "hyperium/hyper": "rust", "hyper": "rust",
    "diesel-rs/diesel": "rust", "diesel": "rust",
    "reqwest-rs/reqwest": "rust", "reqwest": "rust",
    # Go
    "gin-gonic/gin": "go", "gin": "go",
    "go-gorm/gorm": "go", "gorm": "go",
    "spf13/cobra": "go", "cobra": "go",
    "gorilla/mux": "go",
    "gohugoio/hugo": "go", "hugo": "go",
    # Java
    "spring-projects/spring-framework": "java", "spring-framework": "java",
    "junit-team/junit5": "java", "junit5": "java",
    "mockito/mockito": "java", "mockito": "java",
    "fasterxml/jackson-databind": "java", "jackson-databind": "java",
    "google/guava": "java", "guava": "java",
}

# hidden / excluded files: to be hidden from the agent
SKIP_FILES = [
    "run_tests.sh",
    "syn_issue.json",
    "expected_test_output.json",
    "execution_result.json",
    "parsed_commit.json",
    "modified_files.json",
    "modified_entities.json",
    "r2e_tests",
]

SKIP_FILES_NEW = [
    "run_tests.sh",
    "r2e_tests",
]

# # continue msg for agent run loop (in case on null action)
# CONTINUE_MSG = """Please continue working on the task on whatever approach you think is suitable.
# If you think you have solved the task, please first send your answer to user through message and then finish the interaction.
# IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
# IMPORTANT: each response must have both a reasoning and function call. Again each response must have a function call.
# """

CONTINUE_MSG = """
You forgot to use a function call in your response. 
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

# timeout for bash commands
CMD_TIMEOUT = 120  # seconds: 5 minutes
