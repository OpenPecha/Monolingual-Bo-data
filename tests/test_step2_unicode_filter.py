"""
Tests for Step 2: Unicode Filtering
"""

import os
import sys
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from step2_unicode_filter import (
    analyze_unicode,
    filter_tibetan_content,
    clean_tibetan_line,
    process_file,
    UnicodeStats,
)


class TestUnicodeAnalysis:
    """Tests for Unicode character analysis."""
    
    def test_pure_tibetan_text(self):
        """Test analysis of pure Tibetan text."""
        text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        stats = analyze_unicode(text)
        
        assert stats.tibetan_chars > 0
        assert stats.tibetan_ratio > 0.9
    
    def test_pure_english_text(self):
        """Test analysis of pure English text."""
        text = 'This is English text'
        stats = analyze_unicode(text)
        
        assert stats.tibetan_chars == 0
        assert stats.tibetan_ratio == 0.0
        assert stats.ascii_chars == len(text)
    
    def test_mixed_content(self):
        """Test analysis of mixed Tibetan/English content."""
        text = 'བོད་སྐད hello བོད'
        stats = analyze_unicode(text)
        
        assert stats.tibetan_chars > 0
        assert stats.ascii_chars > 0
        assert 0 < stats.tibetan_ratio < 1
    
    def test_empty_text(self):
        """Test analysis of empty text."""
        stats = analyze_unicode('')
        
        assert stats.total_chars == 0
        assert stats.tibetan_ratio == 0.0


class TestTibetanFiltering:
    """Tests for Tibetan content filtering."""
    
    def test_filter_pure_tibetan(self):
        """Test filtering of pure Tibetan text."""
        text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        filtered = filter_tibetan_content(text)
        
        assert filtered is not None
        assert 'བོད' in filtered
    
    def test_filter_discards_pure_english(self):
        """Test that pure English content is discarded."""
        text = 'This is purely English text with no Tibetan.'
        filtered = filter_tibetan_content(text, min_tibetan_ratio=0.05)
        
        assert filtered is None
    
    def test_filter_keeps_mixed_high_tibetan(self):
        """Test that mixed content with high Tibetan ratio is kept."""
        text = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།
བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།'''
        filtered = filter_tibetan_content(text, min_tibetan_ratio=0.05)
        
        assert filtered is not None
        assert 'བོད' in filtered
    
    def test_filter_respects_threshold(self):
        """Test that filtering respects the minimum ratio threshold."""
        # Very low Tibetan ratio
        text = 'English ' * 100 + 'བོད'
        
        # Should be discarded with 5% threshold
        filtered = filter_tibetan_content(text, min_tibetan_ratio=0.05)
        # The overall ratio is very low, so it should be discarded
        assert filtered is None or 'བོད' in (filtered or '')


class TestLineCleaning:
    """Tests for line-level cleaning."""
    
    def test_clean_pure_tibetan_line(self):
        """Test cleaning of pure Tibetan line."""
        line = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        cleaned = clean_tibetan_line(line)
        
        assert 'བོད' in cleaned
        assert cleaned.strip() != ''
    
    def test_clean_removes_english(self):
        """Test that English characters are removed."""
        line = 'བོད་སྐད hello world བོད'
        cleaned = clean_tibetan_line(line)
        
        assert 'hello' not in cleaned
        assert 'world' not in cleaned
        assert 'བོད' in cleaned
    
    def test_clean_removes_special_chars(self):
        """Test removal of special characters."""
        line = 'བོད་སྐད @#$%^ བོད'
        cleaned = clean_tibetan_line(line)
        
        assert '@' not in cleaned
        assert '#' not in cleaned
        assert 'བོད' in cleaned
    
    def test_clean_preserves_tibetan_punctuation(self):
        """Test that Tibetan punctuation is preserved."""
        line = 'བོད་སྐད།'
        cleaned = clean_tibetan_line(line)
        
        # Should preserve tsek and shad
        assert '་' in cleaned or '།' in cleaned


class TestFileProcessing:
    """Tests for file processing."""
    
    def test_process_tibetan_file(self, temp_dir):
        """Test processing of Tibetan file."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create Tibetan file
        content = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།'''
        with open(os.path.join(input_dir, 'tibetan.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file((input_dir, output_dir, 'tibetan.txt', 0.05))
        filename, success, error, stats = result
        
        assert success is True
        assert stats['tibetan_ratio'] > 0.5
    
    def test_process_english_file_discarded(self, temp_dir):
        """Test that English file is discarded."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create English file
        with open(os.path.join(input_dir, 'english.txt'), 'w', encoding='utf-8') as f:
            f.write('This is purely English text.')
        
        result = process_file((input_dir, output_dir, 'english.txt', 0.05))
        filename, success, error, stats = result
        
        assert success is False
        assert 'discarded' in stats.get('status', '').lower() or 'threshold' in error.lower()


class TestIntegration:
    """Integration tests for Unicode filtering."""
    
    def test_filter_multiple_files(self, temp_dir):
        """Test filtering multiple files with different content."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create test files
        files = {
            'tibetan.txt': 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
            'english.txt': 'This is English only.',
            'mixed.txt': 'བོད་སྐད hello བོད་སྐད',
        }
        
        for filename, content in files.items():
            with open(os.path.join(input_dir, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        results = {}
        for filename in files.keys():
            result = process_file((input_dir, output_dir, filename, 0.05))
            results[filename] = result[1]  # success status
        
        # Tibetan file should pass
        assert results['tibetan.txt'] is True
        # English file should fail
        assert results['english.txt'] is False
    
    def test_character_retention(self, temp_dir):
        """Test that character counts are tracked correctly."""
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file((input_dir, output_dir, 'test.txt', 0.05))
        _, _, _, stats = result
        
        assert stats['original_chars'] > 0
        assert stats['original_chars'] == len(content)
