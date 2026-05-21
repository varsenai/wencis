"""
Code Parser — AST-aware semantic chunking using Tree-sitter with graceful fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from silex.utils.logger import setup_logger

log = setup_logger("silex.world.code_parser")

# Tree-sitter optional imports
TREE_SITTER_AVAILABLE = False
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_typescript as tstypescript
    TREE_SITTER_AVAILABLE = True
except ImportError as e:
    log.warning(f"Tree-sitter or grammar packages not found, AST chunking will use fallback. Error: {e}")


@dataclass
class CodeChunk:
    symbol_name: str
    symbol_type: str  # "class" | "function" | "method" | "module" | "other"
    content: str
    start_line: int   # 1-indexed
    end_line: int     # 1-indexed
    file_path: str
    signature: str


class CodeParser:
    """Parses source files into semantically coherent code chunks using AST or fallback."""

    def __init__(self):
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            try:
                self.parsers["python"] = Parser(Language(tspython.language()))
                self.parsers["javascript"] = Parser(Language(tsjavascript.language()))
                self.parsers["typescript"] = Parser(Language(tstypescript.language_typescript()))
                self.parsers["tsx"] = Parser(Language(tstypescript.language_tsx()))
            except Exception as e:
                log.warning(f"Failed to initialize tree-sitter parsers: {e}")

    def parse_file(self, file_path: str, root_dir: str = "") -> List[CodeChunk]:
        """Parse a source file and return a list of semantic CodeChunks."""
        if not os.path.exists(file_path):
            return []

        rel_path = os.path.relpath(file_path, root_dir) if root_dir else file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            log.error(f"Failed to read file {file_path}: {e}")
            return []

        if not content.strip():
            return []

        ext = os.path.splitext(file_path)[1].lower()
        lang_key = self._get_language_key(ext)

        if TREE_SITTER_AVAILABLE and lang_key in self.parsers:
            try:
                return self._parse_ast(content, lang_key, rel_path)
            except Exception as e:
                log.warning(f"AST parsing failed for {rel_path}, falling back: {e}")
                return self._parse_fallback(content, rel_path)
        else:
            return self._parse_fallback(content, rel_path)

    def _get_language_key(self, ext: str) -> Optional[str]:
        if ext == ".py":
            return "python"
        elif ext in (".js", ".jsx", ".mjs", ".cjs"):
            return "javascript"
        elif ext == ".ts":
            return "typescript"
        elif ext == ".tsx":
            return "tsx"
        return None

    def _parse_ast(self, content: str, lang_key: str, rel_path: str) -> List[CodeChunk]:
        """Use tree-sitter to parse source code into AST semantic chunks."""
        parser = self.parsers[lang_key]
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        chunks: List[CodeChunk] = []
        # Keep track of lines covered by specific class/function chunks to build "other" module chunks
        lines_count = len(content.splitlines())
        covered_lines = set()

        def get_node_text(node) -> str:
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

        def get_signature(node, type_name: str) -> str:
            text = get_node_text(node)
            first_line = text.splitlines()[0] if text else ""
            return first_line.strip()

        def visit(node, parent_class: Optional[str] = None):
            node_type = node.type

            # Python types
            is_py_class = (lang_key == "python" and node_type == "class_definition")
            is_py_func = (lang_key == "python" and node_type == "function_definition")

            # JS/TS types
            is_js_class = (lang_key in ("javascript", "typescript", "tsx") and node_type == "class_declaration")
            is_js_func = (lang_key in ("javascript", "typescript", "tsx") and node_type in ("function_declaration", "generator_function_declaration", "method_definition"))

            if is_py_class or is_js_class:
                # Extract class name
                name_node = node.child_by_field_name("name")
                class_name = name_node.text.decode("utf-8", errors="ignore") if name_node else "UnknownClass"
                
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                
                # We cover the class line range
                for line in range(start_line, end_line + 1):
                    covered_lines.add(line)

                # Class chunk
                chunks.append(CodeChunk(
                    symbol_name=class_name,
                    symbol_type="class",
                    content=get_node_text(node),
                    start_line=start_line,
                    end_line=end_line,
                    file_path=rel_path,
                    signature=get_signature(node, "class")
                ))

                # Recursively visit children to find methods inside the class
                for child in node.children:
                    visit(child, parent_class=class_name)

            elif is_py_func or is_js_func:
                name_node = node.child_by_field_name("name")
                func_name = name_node.text.decode("utf-8", errors="ignore") if name_node else "anonymous"
                
                symbol_name = f"{parent_class}.{func_name}" if parent_class else func_name
                symbol_type = "method" if parent_class else "function"
                
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                for line in range(start_line, end_line + 1):
                    covered_lines.add(line)

                chunks.append(CodeChunk(
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    content=get_node_text(node),
                    start_line=start_line,
                    end_line=end_line,
                    file_path=rel_path,
                    signature=get_signature(node, symbol_type)
                ))
                # Do not recurse into functions to avoid extracting inner function definitions as independent chunks

            else:
                for child in node.children:
                    visit(child, parent_class)

        # Start AST visitor
        visit(root)

        # Now group uncovered code blocks into module-level chunks
        lines = content.splitlines()
        current_block = []
        block_start = 1

        for line_idx in range(1, lines_count + 1):
            if line_idx not in covered_lines:
                current_block.append(lines[line_idx - 1])
            else:
                if current_block:
                    block_content = "\n".join(current_block)
                    if block_content.strip():
                        chunks.append(CodeChunk(
                            symbol_name=f"module_level_block_{block_start}",
                            symbol_type="module",
                            content=block_content,
                            start_line=block_start,
                            end_line=line_idx - 1,
                            file_path=rel_path,
                            signature=""
                        ))
                    current_block = []
                block_start = line_idx + 1

        if current_block:
            block_content = "\n".join(current_block)
            if block_content.strip():
                chunks.append(CodeChunk(
                    symbol_name=f"module_level_block_{block_start}",
                    symbol_type="module",
                    content=block_content,
                    start_line=block_start,
                    end_line=lines_count,
                    file_path=rel_path,
                    signature=""
                ))

        # Sort chunks by starting line to maintain file flow
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    def _parse_fallback(self, content: str, rel_path: str) -> List[CodeChunk]:
        """Paragraph-based chunking fallback for non-code files or if tree-sitter is missing."""
        paragraphs = content.split("\n\n")
        chunks: List[CodeChunk] = []
        
        current_chunk_lines = []
        current_char_count = 0
        chunk_start_line = 1
        current_line = 1
        chunk_index = 1

        for p in paragraphs:
            p_lines = p.splitlines()
            p_len = len(p)

            # If adding this paragraph exceeds 1500 characters and we have some lines, flush the current chunk
            if current_char_count + p_len > 1500 and current_chunk_lines:
                block_content = "\n".join(current_chunk_lines)
                chunks.append(CodeChunk(
                    symbol_name=f"chunk_{chunk_index}",
                    symbol_type="other",
                    content=block_content,
                    start_line=chunk_start_line,
                    end_line=current_line - 1,
                    file_path=rel_path,
                    signature=""
                ))
                chunk_index += 1
                current_chunk_lines = []
                current_char_count = 0
                chunk_start_line = current_line

            current_chunk_lines.extend(p_lines)
            current_chunk_lines.append("")  # Re-add paragraph separator
            current_char_count += p_len + 2
            current_line += len(p_lines) + 1

        if current_chunk_lines:
            block_content = "\n".join(current_chunk_lines).strip()
            if block_content:
                chunks.append(CodeChunk(
                    symbol_name=f"chunk_{chunk_index}",
                    symbol_type="other",
                    content=block_content,
                    start_line=chunk_start_line,
                    end_line=current_line - 1,
                    file_path=rel_path,
                    signature=""
                ))

        return chunks
