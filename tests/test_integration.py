"""
Integration Tests for Full Pipeline
====================================
Tests the complete pipeline from raw input to classified output.
"""

import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


class TestFullPipeline:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def pipeline_dirs(self):
        """Create a complete pipeline directory structure."""
        base = tempfile.mkdtemp()
        
        dirs = {
            'raw': os.path.join(base, 'raw'),
            'converted': os.path.join(base, 'converted'),
            'filtered': os.path.join(base, 'filtered'),
            'deduplicated': os.path.join(base, 'deduplicated'),
            'segmented': os.path.join(base, 'segmented'),
            'classified': os.path.join(base, 'classified'),
        }
        
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Create classified subdirs
        for cls in ['A', 'B', 'C']:
            os.makedirs(os.path.join(dirs['classified'], cls), exist_ok=True)
        
        yield dirs
        
        shutil.rmtree(base, ignore_errors=True)
    
    @pytest.fixture
    def sample_tibetan_corpus(self, pipeline_dirs):
        """Create sample Tibetan corpus in raw directory."""
        raw_dir = pipeline_dirs['raw']
        
        # Create multiple documents
        documents = {
            'buddhism.txt': '''ནང་པ་སངས་རྒྱས་པའི་ཆོས་ནི་བོད་ཀྱི་རིག་གཞུང་གི་གཞི་རྩ་རེད།
སངས་རྒྱས་བཅོམ་ལྡན་འདས་ནི་རྒྱ་གར་དུ་སྐུ་འཁྲུངས་པ་རེད།
ཆོས་ལུགས་འདི་བོད་དུ་ལོ་སྟོང་ལྷག་གི་ལོ་རྒྱུས་ཡོད།
དགེ་འདུན་པ་མང་པོས་ཆོས་ཉམས་ལེན་བྱེད་ཀྱི་ཡོད།''',
            
            'geography.txt': '''བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད་པའི་ས་ཁུལ་ཞིག་རེད།
བོད་ཀྱི་རྒྱལ་ས་ནི་ལྷ་ས་རེད།
བོད་ཀྱི་རི་མཐོ་ཤོས་ནི་ཇོ་མོ་གླང་མ་རེད།
བོད་ལ་མཚོ་སྔོན་པོ་ཡང་ཡོད།''',
            
            'language.txt': '''བོད་སྐད་ནི་བོད་པའི་སྐད་ཡིག་རེད།
བོད་ཡིག་ནི་ཐོན་མི་སམ་བྷོ་ཊས་བཟོས་པ་རེད།
བོད་ཡིག་ལ་ཡི་གེ་སུམ་ཅུ་ཡོད།
བོད་མི་མང་པོས་བོད་སྐད་བཤད་ཀྱི་ཡོད།''',
            
            'mixed.txt': '''བོད་སྐད་ནི་བོད་པའི་སྐད་ཡིག་རེད།
This is some English text mixed in.
བོད་ཀྱི་རྒྱལ་ས་ནི་ལྷ་ས་རེད།
More English content here.
བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།''',
            
            'short.txt': '''བོད།
ལྷ།
རི།''',
            
            'english_only.txt': '''This is purely English text.
No Tibetan characters here at all.
Just testing the pipeline.''',
        }
        
        for filename, content in documents.items():
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return documents
    
    def test_step1_to_step2_flow(self, pipeline_dirs, sample_tibetan_corpus):
        """Test flow from Step 1 (conversion) to Step 2 (filtering)."""
        from step1_format_converter import process_file as convert_file
        from step2_unicode_filter import process_file as filter_file
        
        raw_dir = pipeline_dirs['raw']
        converted_dir = pipeline_dirs['converted']
        filtered_dir = pipeline_dirs['filtered']
        
        # Step 1: Convert files
        converted_count = 0
        for filename in os.listdir(raw_dir):
            if filename.endswith('.txt'):
                result = convert_file((raw_dir, converted_dir, filename))
                if result[1]:  # success
                    converted_count += 1
        
        assert converted_count > 0
        
        # Step 2: Filter files
        filtered_count = 0
        for filename in os.listdir(converted_dir):
            if filename.endswith('.txt'):
                result = filter_file((converted_dir, filtered_dir, filename, 0.05))
                if result[1]:  # success
                    filtered_count += 1
        
        # English-only file should be filtered out
        assert filtered_count < converted_count
        
        # Check that filtered files have Tibetan content
        for filename in os.listdir(filtered_dir):
            filepath = os.path.join(filtered_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'བོད' in content or 'ལྷ' in content or any(
                '\u0f00' <= c <= '\u0fff' for c in content
            )
    
    def test_step3_deduplication(self, pipeline_dirs):
        """Test Step 3 deduplication functionality."""
        from step3_deduplicator import compute_minhash, compute_file_hash
        
        filtered_dir = pipeline_dirs['filtered']
        
        # Create duplicate files
        content = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད། བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།'
        
        for i in range(3):
            filepath = os.path.join(filtered_dir, f'duplicate_{i}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Verify exact duplicates are detected
        hashes = []
        for i in range(3):
            filepath = os.path.join(filtered_dir, f'duplicate_{i}.txt')
            hashes.append(compute_file_hash(filepath))
        
        # All hashes should be identical
        assert len(set(hashes)) == 1
        
        # Verify MinHash similarity
        mh1 = compute_minhash(content, num_perm=128)
        mh2 = compute_minhash(content, num_perm=128)
        
        assert mh1.jaccard(mh2) == 1.0
    
    def test_step4_segmentation(self, pipeline_dirs):
        """Test Step 4 segmentation functionality."""
        from step4_segmenter import TibetanSegmenter, filter_sentence, init_worker
        
        init_worker()
        segmenter = TibetanSegmenter(use_botok=False)
        
        # Test text with mixed sentence lengths
        text = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད་པའི་ས་ཁུལ་ཞིག་རེད།'''
        
        sentences = segmenter.segment_sentences(text)
        
        passed = []
        filtered = []
        
        for sent in sentences:
            passes, _ = filter_sentence(sent, segmenter, min_syllables=4, min_tibetan_ratio=0.8)
            if passes:
                passed.append(sent)
            else:
                filtered.append(sent)
        
        # Some sentences should pass, some should be filtered
        assert len(passed) >= 1
    
    def test_step5_classification(self, pipeline_dirs):
        """Test Step 5 classification functionality."""
        from step5_classifier import QualityClassifier, init_worker
        
        import step5_classifier
        step5_classifier._model_path = None
        step5_classifier._threshold_a = 100
        step5_classifier._threshold_b = 500
        
        init_worker()
        
        classifier = QualityClassifier(
            model_path=None,
            threshold_a=100,
            threshold_b=500
        )
        
        sentences = [
            'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
            'བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།',
            'ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།',
        ]
        
        classifications = []
        for sent in sentences:
            class_label, ppl = classifier.classify(sent)
            classifications.append(class_label)
        
        # Without model, all should be class B (default PPL=250)
        assert all(c == 'B' for c in classifications)
    
    def test_pipeline_data_integrity(self, pipeline_dirs, sample_tibetan_corpus):
        """Test that data integrity is maintained through pipeline."""
        from step1_format_converter import process_file as convert_file
        
        raw_dir = pipeline_dirs['raw']
        converted_dir = pipeline_dirs['converted']
        
        # Convert a known file
        filename = 'buddhism.txt'
        original_path = os.path.join(raw_dir, filename)
        
        with open(original_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        result = convert_file((raw_dir, converted_dir, filename))
        
        assert result[1] is True  # success
        
        # Check converted content
        converted_path = os.path.join(converted_dir, filename)
        with open(converted_path, 'r', encoding='utf-8') as f:
            converted_content = f.read()
        
        # Content should be identical for txt files
        assert converted_content == original_content
    
    def test_pipeline_statistics_tracking(self, pipeline_dirs, sample_tibetan_corpus):
        """Test that statistics are tracked correctly."""
        from step1_format_converter import process_file as convert_file
        
        raw_dir = pipeline_dirs['raw']
        converted_dir = pipeline_dirs['converted']
        
        total_files = len([f for f in os.listdir(raw_dir) if f.endswith('.txt')])
        successful = 0
        
        for filename in os.listdir(raw_dir):
            if filename.endswith('.txt'):
                result = convert_file((raw_dir, converted_dir, filename))
                if result[1]:
                    successful += 1
        
        # Check that we processed all files
        assert successful == total_files


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_input_directory(self, temp_dir):
        """Test handling of empty input directory."""
        from step1_format_converter import get_supported_files
        
        empty_dir = os.path.join(temp_dir, 'empty')
        os.makedirs(empty_dir, exist_ok=True)
        
        files = get_supported_files(empty_dir)
        
        assert len(files) == 0
    
    def test_unicode_edge_cases(self, temp_dir):
        """Test handling of various Unicode edge cases."""
        from step2_unicode_filter import analyze_unicode, filter_tibetan_content
        
        # Test with combining characters
        text_with_combining = 'བོད་སྐད་\u0f71\u0f72'  # With subjoined vowels
        stats = analyze_unicode(text_with_combining)
        
        assert stats.tibetan_chars > 0
        
        # Test with mixed scripts
        mixed = 'བོད་སྐད 中文 한국어'
        stats = analyze_unicode(mixed)
        
        assert stats.tibetan_chars > 0
        assert stats.other_chars > 0
    
    def test_very_long_document(self, temp_dir):
        """Test handling of very long documents."""
        from step3_deduplicator import compute_minhash, get_tibetan_shingles
        
        # Create a long document by repeating text
        base_text = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད། '
        long_text = base_text * 1000  # Repeat 1000 times
        
        # Should handle without error
        shingles = get_tibetan_shingles(long_text)
        minhash = compute_minhash(long_text)
        
        assert len(shingles) > 0
        assert minhash is not None
    
    def test_special_tibetan_characters(self, temp_dir):
        """Test handling of special Tibetan characters."""
        from step2_unicode_filter import analyze_unicode
        
        # Text with various Tibetan symbols
        special_chars = 'ༀ༁༂༃༄༅༆༇༈༉༊་༌།༎༏༐༑༒༓༔༕༖༗༘༙'
        stats = analyze_unicode(special_chars)
        
        # All should be recognized as Tibetan
        assert stats.tibetan_ratio > 0.9


class TestConcurrency:
    """Tests for concurrent processing."""
    
    def test_parallel_file_processing(self, temp_dir):
        """Test that parallel processing works correctly."""
        from multiprocessing import Pool
        from step1_format_converter import process_file
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create multiple test files
        for i in range(10):
            filepath = os.path.join(input_dir, f'file_{i}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f'བོད་སྐད་ཡིག་གྲངས་{i}།')
        
        # Process in parallel
        files = [f'file_{i}.txt' for i in range(10)]
        args = [(input_dir, output_dir, f) for f in files]
        
        with Pool(processes=4) as pool:
            results = pool.map(process_file, args)
        
        # All should succeed
        successes = sum(1 for r in results if r[1])
        assert successes == 10
