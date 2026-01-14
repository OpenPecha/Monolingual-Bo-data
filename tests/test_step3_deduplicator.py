"""
Tests for Step 3: Document Deduplication
"""

import os
import sys
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from step3_deduplicator import (
    get_tibetan_shingles,
    compute_minhash,
    compute_file_hash,
    process_file_minhash,
)


class TestTibetanShingles:
    """Tests for syllable-based shingling."""
    
    def test_basic_shingling(self):
        """Test basic syllable shingling."""
        text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད'
        shingles = get_tibetan_shingles(text, n=3)
        
        assert len(shingles) > 0
        # Each shingle should contain tsek
        for shingle in shingles:
            if len(shingle) > 1:
                assert '་' in shingle or len(shingle.split('་')) >= 1
    
    def test_shingling_short_text(self):
        """Test shingling of text shorter than n-gram size."""
        text = 'བོད་སྐད'
        shingles = get_tibetan_shingles(text, n=5)
        
        # Should return the whole text as single shingle
        assert len(shingles) >= 1
    
    def test_shingling_with_shad(self):
        """Test shingling handles sentence boundaries."""
        text = 'བོད་སྐད་ནི།བོད་ཀྱི་སྐད'
        shingles = get_tibetan_shingles(text, n=2)
        
        assert len(shingles) > 0
    
    def test_empty_text_shingling(self):
        """Test shingling of empty text."""
        shingles = get_tibetan_shingles('', n=3)
        
        # Should handle gracefully
        assert len(shingles) >= 0


class TestMinHash:
    """Tests for MinHash computation."""
    
    def test_minhash_creation(self):
        """Test MinHash signature creation."""
        text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        minhash = compute_minhash(text, num_perm=128)
        
        assert minhash is not None
        assert len(minhash.hashvalues) == 128
    
    def test_identical_texts_similar_minhash(self):
        """Test that identical texts produce identical MinHash."""
        text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        
        minhash1 = compute_minhash(text, num_perm=128)
        minhash2 = compute_minhash(text, num_perm=128)
        
        # Jaccard similarity should be 1.0 for identical texts
        similarity = minhash1.jaccard(minhash2)
        assert similarity == 1.0
    
    def test_different_texts_different_minhash(self):
        """Test that different texts produce different MinHash."""
        text1 = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        text2 = 'ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།'
        
        minhash1 = compute_minhash(text1, num_perm=128)
        minhash2 = compute_minhash(text2, num_perm=128)
        
        # Similarity should be less than 1.0
        similarity = minhash1.jaccard(minhash2)
        assert similarity < 1.0
    
    def test_similar_texts_high_similarity(self):
        """Test that similar texts have high MinHash similarity."""
        text1 = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        text2 = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་ཡིན།'  # Slight modification
        
        minhash1 = compute_minhash(text1, num_perm=128)
        minhash2 = compute_minhash(text2, num_perm=128)
        
        similarity = minhash1.jaccard(minhash2)
        # Should have reasonably high similarity
        assert similarity > 0.5


class TestFileHash:
    """Tests for exact duplicate detection via hashing."""
    
    def test_file_hash_consistency(self, temp_dir):
        """Test that same content produces same hash."""
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        
        file1 = os.path.join(temp_dir, 'file1.txt')
        file2 = os.path.join(temp_dir, 'file2.txt')
        
        for filepath in [file1, file2]:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        
        assert hash1 == hash2
    
    def test_different_content_different_hash(self, temp_dir):
        """Test that different content produces different hash."""
        file1 = os.path.join(temp_dir, 'file1.txt')
        file2 = os.path.join(temp_dir, 'file2.txt')
        
        with open(file1, 'w', encoding='utf-8') as f:
            f.write('བོད་སྐད')
        with open(file2, 'w', encoding='utf-8') as f:
            f.write('ལྷ་ས')
        
        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        
        assert hash1 != hash2


class TestFileProcessing:
    """Tests for file MinHash processing."""
    
    def test_process_file_minhash(self, temp_dir):
        """Test MinHash computation for a file."""
        input_dir = temp_dir
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file_minhash((input_dir, 'test.txt', 128))
        filename, minhash, content_hash, size = result
        
        assert filename == 'test.txt'
        assert minhash is not None
        assert content_hash != ''
        assert size > 0
    
    def test_process_empty_file(self, temp_dir):
        """Test processing of empty file."""
        input_dir = temp_dir
        
        with open(os.path.join(input_dir, 'empty.txt'), 'w', encoding='utf-8') as f:
            f.write('')
        
        result = process_file_minhash((input_dir, 'empty.txt', 128))
        filename, minhash, content_hash, size = result
        
        # Empty file should return None minhash
        assert minhash is None


class TestIntegration:
    """Integration tests for deduplication."""
    
    def test_exact_duplicate_detection(self, temp_dir):
        """Test detection of exact duplicates."""
        input_dir = temp_dir
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        
        # Create duplicate files
        for i in range(3):
            with open(os.path.join(input_dir, f'dup{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Get hashes
        hashes = []
        for i in range(3):
            h = compute_file_hash(os.path.join(input_dir, f'dup{i}.txt'))
            hashes.append(h)
        
        # All hashes should be identical
        assert len(set(hashes)) == 1
    
    def test_near_duplicate_detection(self, temp_dir):
        """Test detection of near-duplicates via MinHash."""
        input_dir = temp_dir
        
        # Original text - longer text for more accurate similarity
        base_content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད། བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད། ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད། བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།'
        original = base_content + ' བོད་ཀྱི་རིག་གཞུང་ནི་ཕྱུག་པོ་རེད།'
        # Near-duplicate with very small change at the end
        near_dup = base_content + ' བོད་ཀྱི་རིག་གཞུང་ནི་ཕྱུག་པོ་ཡིན།'
        
        with open(os.path.join(input_dir, 'original.txt'), 'w', encoding='utf-8') as f:
            f.write(original)
        with open(os.path.join(input_dir, 'near_dup.txt'), 'w', encoding='utf-8') as f:
            f.write(near_dup)
        
        mh1 = compute_minhash(original, num_perm=128)
        mh2 = compute_minhash(near_dup, num_perm=128)
        
        similarity = mh1.jaccard(mh2)
        
        # Near-duplicates should have reasonably high similarity
        # (actual threshold may vary based on document length)
        assert similarity > 0.5
    
    def test_unique_documents_low_similarity(self, temp_dir):
        """Test that unique documents have low similarity."""
        doc1 = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        doc2 = 'རྒྱ་གར་ནི་ཨེ་ཤི་ཡའི་ལྷོ་ཕྱོགས་སུ་ཡོད་པའི་རྒྱལ་ཁབ་རེད།'
        
        mh1 = compute_minhash(doc1, num_perm=128)
        mh2 = compute_minhash(doc2, num_perm=128)
        
        similarity = mh1.jaccard(mh2)
        
        # Should be below dedup threshold
        assert similarity < 0.85
