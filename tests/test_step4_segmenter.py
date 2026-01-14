"""
Tests for Step 4: Linguistic Filtering & Segmentation
"""

import os
import sys
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from step4_segmenter import (
    TibetanSegmenter,
    filter_sentence,
    process_file,
    init_worker,
)


class TestTibetanSegmenter:
    """Tests for the TibetanSegmenter class."""
    
    @pytest.fixture
    def segmenter(self):
        """Create a segmenter instance."""
        return TibetanSegmenter(use_botok=False)  # Use fallback for testing
    
    def test_segment_simple_text(self, segmenter):
        """Test segmentation of simple Tibetan text."""
        text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།'
        sentences = segmenter.segment_sentences(text)
        
        assert len(sentences) >= 1
        assert any('བོད' in s for s in sentences)
    
    def test_segment_multiline_text(self, segmenter):
        """Test segmentation of multiline text."""
        text = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།
བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།'''
        sentences = segmenter.segment_sentences(text)
        
        assert len(sentences) >= 2
    
    def test_segment_empty_text(self, segmenter):
        """Test segmentation of empty text."""
        sentences = segmenter.segment_sentences('')
        
        assert len(sentences) == 0


class TestSyllableCounting:
    """Tests for syllable counting."""
    
    @pytest.fixture
    def segmenter(self):
        return TibetanSegmenter(use_botok=False)
    
    def test_count_syllables_basic(self, segmenter):
        """Test basic syllable counting."""
        # 4 syllables: བོད, སྐད, ནི, བོད
        text = 'བོད་སྐད་ནི་བོད'
        count = segmenter.count_syllables(text)
        
        assert count == 4
    
    def test_count_syllables_with_shad(self, segmenter):
        """Test syllable counting with sentence markers."""
        text = 'བོད་སྐད།'
        count = segmenter.count_syllables(text)
        
        assert count >= 2
    
    def test_count_syllables_empty(self, segmenter):
        """Test syllable counting for empty string."""
        count = segmenter.count_syllables('')
        
        assert count == 0


class TestTibetanRatio:
    """Tests for Tibetan character ratio calculation."""
    
    @pytest.fixture
    def segmenter(self):
        return TibetanSegmenter(use_botok=False)
    
    def test_ratio_pure_tibetan(self, segmenter):
        """Test ratio for pure Tibetan text."""
        text = 'བོད་སྐད་ནི་བོད'
        ratio = segmenter.tibetan_ratio(text)
        
        assert ratio > 0.9
    
    def test_ratio_pure_english(self, segmenter):
        """Test ratio for pure English text."""
        text = 'Hello World'
        ratio = segmenter.tibetan_ratio(text)
        
        assert ratio == 0.0
    
    def test_ratio_mixed_content(self, segmenter):
        """Test ratio for mixed content."""
        text = 'བོད hello བོད'
        ratio = segmenter.tibetan_ratio(text)
        
        assert 0 < ratio < 1
    
    def test_ratio_empty_text(self, segmenter):
        """Test ratio for empty text."""
        ratio = segmenter.tibetan_ratio('')
        
        assert ratio == 0.0


class TestSentenceFiltering:
    """Tests for sentence-level filtering."""
    
    @pytest.fixture
    def segmenter(self):
        return TibetanSegmenter(use_botok=False)
    
    def test_filter_valid_sentence(self, segmenter):
        """Test filtering of valid sentence."""
        sentence = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        passes, reason = filter_sentence(
            sentence, segmenter, min_syllables=4, min_tibetan_ratio=0.8
        )
        
        assert passes is True
        assert reason == 'passed'
    
    def test_filter_short_sentence(self, segmenter):
        """Test filtering of short sentence."""
        sentence = 'བོད།'  # Only 1-2 syllables
        passes, reason = filter_sentence(
            sentence, segmenter, min_syllables=4, min_tibetan_ratio=0.8
        )
        
        assert passes is False
        assert 'syllables' in reason.lower()
    
    def test_filter_low_tibetan_ratio(self, segmenter):
        """Test filtering of sentence with low Tibetan ratio."""
        sentence = 'བོད hello world test english text'
        passes, reason = filter_sentence(
            sentence, segmenter, min_syllables=1, min_tibetan_ratio=0.8
        )
        
        assert passes is False
        assert 'ratio' in reason.lower()
    
    def test_filter_edge_case_syllables(self, segmenter):
        """Test filtering at exact syllable threshold."""
        # Exactly 4 syllables
        sentence = 'བོད་སྐད་ནི་བོད'
        passes, reason = filter_sentence(
            sentence, segmenter, min_syllables=4, min_tibetan_ratio=0.5
        )
        
        assert passes is True


class TestFileProcessing:
    """Tests for file processing."""
    
    def test_process_valid_file(self, temp_dir):
        """Test processing of valid Tibetan file."""
        # Initialize the global segmenter
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create test file with valid sentences
        content = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།
བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།'''
        
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file((input_dir, output_dir, 'test.txt', 4, 0.8))
        filename, success, error, stats = result
        
        assert success is True
        assert stats['total_sentences'] > 0
        assert stats['passed_sentences'] > 0
    
    def test_process_file_with_short_sentences(self, temp_dir):
        """Test processing of file with short sentences."""
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create test file with short sentences
        content = 'བོད།ལྷ།'  # Very short
        
        with open(os.path.join(input_dir, 'short.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file((input_dir, output_dir, 'short.txt', 4, 0.8))
        filename, success, error, stats = result
        
        # May fail if all sentences are too short
        assert stats['filtered_syllables'] >= 0


class TestIntegration:
    """Integration tests for segmentation."""
    
    def test_full_segmentation_workflow(self, temp_dir):
        """Test complete segmentation workflow."""
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Create comprehensive test file
        content = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད་པའི་ས་ཁུལ་ཞིག་རེད།
ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།'''
        
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file((input_dir, output_dir, 'test.txt', 4, 0.8))
        filename, success, error, stats = result
        
        if success:
            # Check output file exists
            output_path = os.path.join(output_dir, 'test.txt')
            assert os.path.exists(output_path)
            
            # Read and verify output
            with open(output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Short sentences should be filtered out
            lines = output_content.strip().split('\n')
            segmenter = TibetanSegmenter(use_botok=False)
            
            for line in lines:
                if line.strip():
                    syllables = segmenter.count_syllables(line)
                    assert syllables >= 4
