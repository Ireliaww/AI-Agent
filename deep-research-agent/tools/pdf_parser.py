"""
PDF Parser for Academic Papers

Extracts structured content from academic papers including:
- Title, authors, abstract
- Sections and subsections
- Mathematical equations
- References
- Key content for LLM analysis
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PaperContent:
    """Structured content extracted from an academic paper"""
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    equations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    full_text: str = ""
    
    def get_summary(self) -> str:
        """Get a concise summary for LLM input"""
        return f"""Title: {self.title}
Authors: {', '.join(self.authors)}

Abstract:
{self.abstract}

Sections:
{self._format_sections()}
"""
    
    def _format_sections(self) -> str:
        """Format sections for readable output"""
        formatted = []
        for section_name, content in self.sections.items():
            # Truncate very long sections
            truncated = content[:500] + "..." if len(content) > 500 else content
            formatted.append(f"## {section_name}\n{truncated}")
        return "\n\n".join(formatted)


class PDFParser:
    """
    Parse academic papers from PDF files
    
    Features:
    - Extract metadata (title, authors, abstract)
    - Identify section structure
    - Extract mathematical equations
    - Parse references
    """
    
    def __init__(self):
        self.section_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[IVX]+\.?\s+[A-Z]',  # Roman numerals
            r'^(Abstract|Introduction|Related Work|Methodology|Experiments|Results|Conclusion|References)',
        ]
    
    def parse(self, pdf_path: str) -> PaperContent:
        """
        Parse a PDF paper
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PaperContent with extracted information
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Extract all text
            full_text = self._extract_text(doc)
            
            # Extract metadata
            title = self._extract_title(doc, full_text)
            authors = self._extract_authors(full_text)
            abstract = self._extract_abstract(full_text)
            
            # Extract structure
            sections = self._extract_sections(full_text)
            equations = self._extract_equations(full_text)
            references = self._extract_references(full_text)
            
            doc.close()
            
            return PaperContent(
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                equations=equations,
                references=references,
                full_text=full_text
            )
        
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")
    
    def _extract_text(self, doc: fitz.Document) -> str:
        """Extract all text from PDF"""
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    
    def _extract_title(self, doc: fitz.Document, text: str) -> str:
        """Extract paper title"""
        # Try PDF metadata first
        metadata = doc.metadata
        if metadata and metadata.get('title'):
            return metadata['title']
        
        # Fallback: first line often contains title
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and not line.isupper():  # Avoid headers
                return line
        
        return "Unknown Title"
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names"""
        # Simple heuristic: look for lines after title before abstract
        lines = text.split('\n')
        authors = []
        
        # Look for lines that might contain authors
        # Usually after title and before abstract
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            # Check if line contains potential author names
            if re.match(r'^[A-Z][a-z]+.*[A-Z][a-z]+', line) and '@' not in line:
                # Split by common separators
                potential_authors = re.split(r'[,;]|\sand\s', line)
                authors.extend([a.strip() for a in potential_authors if a.strip()])
        
        return authors[:10] if authors else []  # Limit to reasonable number
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section"""
        # Look for "Abstract" header
        abstract_pattern = r'(?:Abstract|ABSTRACT)\s*\n(.*?)(?:\n\s*\n|\n(?:1\.?\s+Introduction|Introduction))'
        match = re.search(abstract_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            abstract = match.group(1).strip()
            # Clean up
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract
        
        return ""
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract main sections"""
        sections = {}
        
        # Common section names
        section_names = [
            'Introduction', 'Related Work', 'Background', 
            'Methodology', 'Method', 'Approach',
            'Experiments', 'Experimental Setup', 'Results',
            'Discussion', 'Conclusion', 'Future Work'
        ]
        
        for section_name in section_names:
            # Try to find section
            pattern = rf'(?:^|\n)\s*(?:\d+\.?\s+)?{section_name}\s*\n(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:{"|".join(section_names)})|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                # Clean up whitespace
                content = re.sub(r'\s+', ' ', content)
                sections[section_name] = content[:1000]  # Limit length
        
        return sections
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations (basic)"""
        # Look for LaTeX-style equations
        equations = []
        
        # Pattern for inline math: $...$
        inline_pattern = r'\$([^$]+)\$'
        equations.extend(re.findall(inline_pattern, text))
        
        # Pattern for display math: $$...$$
        display_pattern = r'\$\$([^$]+)\$\$'
        equations.extend(re.findall(display_pattern, text))
        
        return equations[:20]  # Limit to first 20 equations
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references section"""
        # Look for References section
        ref_pattern = r'(?:References|REFERENCES|Bibliography)\s*\n(.*?)(?:\n\s*$|$)'
        match = re.search(ref_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            ref_text = match.group(1)
            # Split into individual references
            # Common pattern: [1] or 1. at start of line
            refs = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', ref_text)
            return [ref.strip() for ref in refs if ref.strip()][:30]  # Limit to 30
        
        return []


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        parser = PDFParser()
        
        print("Parsing PDF...")
        paper = parser.parse(pdf_path)
        
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        print(f"\nAbstract:\n{paper.abstract[:200]}...")
        print(f"\nSections found: {list(paper.sections.keys())}")
        print(f"Equations found: {len(paper.equations)}")
        print(f"References found: {len(paper.references)}")
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
