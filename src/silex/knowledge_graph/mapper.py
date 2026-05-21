"""
Lightweight Skeleton Mapper (World Model V2)

Parses Python and TypeScript/JavaScript files to extract deterministic dependencies
(Imports, Functions, Classes) without injecting heavy raw ASTs into the LLM context.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from silex.utils.logger import setup_logger

log = setup_logger("silex.world_model.mapper")


class SkeletonMapper:
    """Extracts structural metadata from source code safely and deterministically."""

    @staticmethod
    def parse_python_file(filepath: Path) -> dict:
        """Use Python's built-in AST to extract structure. No C-extensions needed."""
        try:
            if filepath.stat().st_size > 512 * 1024:
                log.warning(f"Skipping {filepath.name} — exceeds mapper size limit (512KB).")
                return {"error": "File too large", "file": str(filepath)}
                
            content = filepath.read_text(encoding="utf-8")
            tree = ast.parse(content)
            
            imports = []
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
            return {
                "file": str(filepath),
                "type": "python",
                "imports": list(set(imports)),
                "functions": list(set(functions)),
                "classes": list(set(classes)),
            }
        except Exception as e:
            log.warning(f"Failed to map Python file {filepath}: {e}")
            return {"error": str(e), "file": str(filepath)}

    @staticmethod
    def parse_typescript_file(filepath: Path) -> dict:
        """
        Use fast Regex for TS/JS files to avoid heavy Node dependencies 
        or tree-sitter Windows compile friction.
        """
        try:
            if filepath.stat().st_size > 512 * 1024:
                log.warning(f"Skipping {filepath.name} — exceeds mapper size limit (512KB).")
                return {"error": "File too large", "file": str(filepath)}
                
            content = filepath.read_text(encoding="utf-8")
            
            # Match ES6 imports: from 'module' or from "module"
            import_pattern = r'from\s+[\'"]([^\'"]+)[\'"]'
            # Match require('module')
            require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
            
            # Match functions: function foo() or const foo = () => 
            func_pattern = r'(?:function\s+(\w+))|(?:const\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>)'
            # Match classes: class Foo
            class_pattern = r'class\s+(\w+)'
            
            imports = re.findall(import_pattern, content) + re.findall(require_pattern, content)
            
            functions = []
            for match in re.finditer(func_pattern, content):
                funcs = [g for g in match.groups() if g]
                if funcs:
                    functions.append(funcs[0])
                    
            classes = re.findall(class_pattern, content)
            
            return {
                "file": str(filepath),
                "type": "typescript",
                "imports": list(set(imports)),
                "functions": list(set(functions)),
                "classes": list(set(classes)),
            }
        except Exception as e:
            log.warning(f"Failed to map TS/JS file {filepath}: {e}")
            return {"error": str(e), "file": str(filepath)}

    @classmethod
    def map_file(cls, filepath: Path) -> dict | None:
        """Determine file type and route to the correct parser."""
        # Issue 10: Prevent Symlink Path Traversal
        if filepath.is_symlink():
            log.debug(f"Skipping symlink to prevent traversal: {filepath.name}")
            return None
            
        if filepath.suffix == ".py":
            return cls.parse_python_file(filepath)
        elif filepath.suffix in {".js", ".jsx", ".ts", ".tsx"}:
            return cls.parse_typescript_file(filepath)
        return None
