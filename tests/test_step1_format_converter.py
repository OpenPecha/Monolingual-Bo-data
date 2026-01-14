"""
Tests for Step 1: File Format Conversion
"""

import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from step1_format_converter import (
    detect_encoding,
    convert_txt,
    validate_utf8,
    process_file,
    get_supported_files,
)


class TestEncodingDetection:
    """Tests for encoding detection."""
    
    def test_detect_utf8_encoding(self, temp_dir):
        """Test detection of UTF-8 encoded files."""
        filepath = os.path.join(temp_dir, 'utf8.txt')
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        encoding = detect_encoding(filepath)
        assert encoding.lower() in ['utf-8', 'utf8', 'ascii']
    
    def test_detect_ascii_encoding(self, temp_dir):
        """Test detection of ASCII encoded files."""
        filepath = os.path.join(temp_dir, 'ascii.txt')
        content = 'Hello World'
        
        with open(filepath, 'w', encoding='ascii') as f:
            f.write(content)
        
        encoding = detect_encoding(filepath)
        assert encoding.lower() in ['ascii', 'utf-8', 'utf8']


class TestTextConversion:
    """Tests for text file conversion."""
    
    def test_convert_utf8_txt(self, sample_tibetan_file):
        """Test conversion of UTF-8 text file."""
        content = convert_txt(sample_tibetan_file)
        
        assert content is not None
        assert 'བོད' in content
        assert len(content) > 0
    
    def test_convert_preserves_tibetan(self, temp_dir):
        """Test that conversion preserves Tibetan characters."""
        filepath = os.path.join(temp_dir, 'tibetan.txt')
        original = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(original)
        
        converted = convert_txt(filepath)
        assert converted == original


class TestUTF8Validation:
    """Tests for UTF-8 validation."""
    
    def test_valid_utf8(self):
        """Test validation of valid UTF-8 strings."""
        valid_strings = [
            'Hello World',
            'བོད་སྐད།',
            '混合中文',
            '🎉 Emoji test',
        ]
        
        for s in valid_strings:
            assert validate_utf8(s) is True
    
    def test_validates_tibetan_text(self):
        """Test validation of Tibetan text."""
        tibetan = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        assert validate_utf8(tibetan) is True


class TestFileProcessing:
    """Tests for file processing."""
    
    def test_process_txt_file(self, temp_dir):
        """Test processing of a .txt file."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create test file
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Process file
        result = process_file((input_dir, output_dir, 'test.txt'))
        filename, success, error = result
        
        assert success is True
        assert error == ''
        assert os.path.exists(os.path.join(output_dir, 'test.txt'))
    
    def test_process_empty_file(self, temp_dir):
        """Test processing of an empty file."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create empty file
        with open(os.path.join(input_dir, 'empty.txt'), 'w', encoding='utf-8') as f:
            f.write('')
        
        result = process_file((input_dir, output_dir, 'empty.txt'))
        filename, success, error = result
        
        assert success is False
        assert 'Empty' in error or 'empty' in error.lower()


class TestSupportedFiles:
    """Tests for supported file detection."""
    
    def test_get_txt_files(self, temp_dir):
        """Test detection of .txt files."""
        # Create test files
        files = ['test1.txt', 'test2.txt', 'test.pdf', 'test.doc']
        for f in files:
            with open(os.path.join(temp_dir, f), 'w') as fp:
                fp.write('test')
        
        supported = get_supported_files(temp_dir)
        
        assert 'test1.txt' in supported
        assert 'test2.txt' in supported
    
    def test_ignores_unsupported_formats(self, temp_dir):
        """Test that unsupported formats are found but may not convert."""
        # Create unsupported file
        with open(os.path.join(temp_dir, 'test.xyz'), 'w') as f:
            f.write('test')
        
        supported = get_supported_files(temp_dir)
        
        # .xyz should not be in supported list
        assert 'test.xyz' not in supported


class TestIntegration:
    """Integration tests for format conversion."""
    
    def test_end_to_end_conversion(self, temp_dir):
        """Test complete conversion workflow."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create multiple test files
        test_content = {
            'file1.txt': 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
            'file2.txt': 'བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།',
        }
        
        for filename, content in test_content.items():
            with open(os.path.join(input_dir, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Process each file
        for filename in test_content.keys():
            result = process_file((input_dir, output_dir, filename))
            assert result[1] is True  # success
        
        # Verify outputs
        for filename in test_content.keys():
            output_path = os.path.join(output_dir, filename)
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'བོད' in content
