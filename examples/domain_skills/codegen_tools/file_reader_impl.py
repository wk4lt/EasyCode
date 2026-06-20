"""File reader skill implementation.

Reads files from the local filesystem and returns their contents.

Layer: Skill layer (first layer).
"""

from pathlib import Path

from liteagent.core.base_skill import BaseSkill


class FileReaderImpl(BaseSkill):
    """Read file contents from the local filesystem."""

    MAX_SIZE = 1 * 1024 * 1024

    ALLOWED_EXTENSIONS = {".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml"}

    def execute(self, file_path: str) -> dict:
        """Read a file and return its contents.

        Args:
            file_path: Path to the file to read.

        Returns:
            dict with 'status' and 'content' or 'error'.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"status": "error", "error": f"File not found: {file_path}"}

            if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                return {
                    "status": "error",
                    "error": f"Unsupported file type '{path.suffix}'. Allowed: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}",
                }

            if path.stat().st_size > self.MAX_SIZE:
                return {
                    "status": "error",
                    "error": f"File too large ({path.stat().st_size} bytes). Max: {self.MAX_SIZE} bytes.",
                }

            content = path.read_text(encoding="utf-8")
            return {
                "status": "ok",
                "file_path": str(path),
                "file_name": path.name,
                "content": content,
                "size_bytes": len(content.encode("utf-8")),
            }

        except UnicodeDecodeError:
            return {"status": "error", "error": f"File is not valid UTF-8: {file_path}"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to read file: {e}"}
