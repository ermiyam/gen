from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from .response_handler import SpecializedModel, ModelConfig

logger = logging.getLogger(__name__)

class CodeGenModel(SpecializedModel):
    """Specialized model for code generation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.code_patterns = {
            'function': r'def\s+\w+\s*\([^)]*\)\s*:',
            'class': r'class\s+\w+\s*(?:\([^)]*\))?\s*:',
            'import': r'(?:from\s+[\w.]+\s+)?import\s+[\w.]+(?:\s+as\s+\w+)?'
        }
    
    def initialize(self):
        """Initialize the code generation model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            logger.info(f"Code generation model initialized: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize code generation model: {e}")
            raise
    
    def predict(self, input_data: str) -> Dict[str, Any]:
        """Generate code based on input specification"""
        try:
            # Prepare the prompt
            prompt = self._prepare_code_prompt(input_data)
            
            # Generate code
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            # Process the generated code
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and validate code
            code_blocks = self._extract_code_blocks(generated_code)
            validated_code = self._validate_code(code_blocks)
            
            return {
                'code': validated_code,
                'confidence': self._calculate_confidence(validated_code),
                'suggestions': self._generate_suggestions(validated_code)
            }
            
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return {
                'code': '',
                'error': str(e),
                'confidence': 0.0
            }
    
    def _prepare_code_prompt(self, input_spec: str) -> str:
        """Prepare a prompt for code generation"""
        return f"""
Generate Python code based on the following specification:
{input_spec}

Requirements:
- Follow PEP 8 style guidelines
- Include proper error handling
- Add descriptive comments
- Use type hints where appropriate

Code:
"""

    def _extract_code_blocks(self, text: str) -> str:
        """Extract code blocks from generated text"""
        import re
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        return text
    
    def _validate_code(self, code: str) -> str:
        """Validate generated code"""
        # Basic validation
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax error: {e}")
            # Try to fix common issues
            fixed_code = self._fix_common_issues(code)
            return fixed_code
    
    def _fix_common_issues(self, code: str) -> str:
        """Fix common code generation issues"""
        # Remove incomplete lines
        lines = code.split('\n')
        complete_lines = [line for line in lines if not line.strip().endswith((':', '(', '{', '['))]
        
        # Ensure proper indentation
        fixed_lines = []
        indent_level = 0
        for line in complete_lines:
            if line.strip():
                if any(line.strip().startswith(word) for word in ['else:', 'elif', 'except:', 'finally:']):
                    indent_level = max(0, indent_level - 1)
                fixed_lines.append('    ' * indent_level + line.lstrip())
                if line.strip().endswith(':'):
                    indent_level += 1
        
        return '\n'.join(fixed_lines)
    
    def _calculate_confidence(self, code: str) -> float:
        """Calculate confidence score for generated code"""
        score = 1.0
        
        # Check for basic code patterns
        for pattern in self.code_patterns.values():
            if not re.search(pattern, code):
                score *= 0.9
        
        # Check for common issues
        if 'pass' in code:
            score *= 0.8
        if '# TODO' in code:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def _generate_suggestions(self, code: str) -> list[str]:
        """Generate improvement suggestions for the code"""
        suggestions = []
        
        # Check for type hints
        if 'def ' in code and ': ' not in code:
            suggestions.append("Consider adding type hints to function parameters")
        
        # Check for docstrings
        if 'def ' in code and '"""' not in code:
            suggestions.append("Add docstrings to functions for better documentation")
        
        # Check for error handling
        if 'try' not in code:
            suggestions.append("Consider adding error handling with try-except blocks")
        
        return suggestions 