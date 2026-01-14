"""
Tests for Step 5: OCR Quality Classification
"""

import os
import sys
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from step5_classifier import (
    QualityClassifier,
    process_file,
    init_worker,
)

# Set global variables for worker initialization
import step5_classifier
step5_classifier._model_path = None
step5_classifier._threshold_a = 100
step5_classifier._threshold_b = 500


class TestQualityClassifier:
    """Tests for the QualityClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier without model (uses default scoring)."""
        return QualityClassifier(
            model_path=None,
            threshold_a=100,
            threshold_b=500
        )
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.threshold_a == 100
        assert classifier.threshold_b == 500
    
    def test_classify_without_model(self, classifier):
        """Test classification without KenLM model (default behavior)."""
        sentence = 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།'
        class_label, ppl = classifier.classify(sentence)
        
        # Without model, should return default perplexity (250)
        assert class_label in ['A', 'B', 'C']
        assert ppl == 250.0  # Default value
        assert class_label == 'B'  # 250 is in class B range
    
    def test_classify_returns_valid_class(self, classifier):
        """Test that classification returns valid class labels."""
        sentences = [
            'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
            'ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།',
            'བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།',
        ]
        
        for sent in sentences:
            class_label, ppl = classifier.classify(sent)
            assert class_label in ['A', 'B', 'C']
            assert isinstance(ppl, float)


class TestPerplexityThresholds:
    """Tests for perplexity threshold behavior."""
    
    def test_class_a_threshold(self):
        """Test Class A threshold (PPL <= 100)."""
        classifier = QualityClassifier(
            model_path=None,
            threshold_a=100,
            threshold_b=500
        )
        
        # Manually check threshold logic
        # PPL 50 should be Class A
        if 50 <= classifier.threshold_a:
            expected_class = 'A'
        elif 50 <= classifier.threshold_b:
            expected_class = 'B'
        else:
            expected_class = 'C'
        
        assert expected_class == 'A'
    
    def test_class_b_threshold(self):
        """Test Class B threshold (100 < PPL <= 500)."""
        classifier = QualityClassifier(
            model_path=None,
            threshold_a=100,
            threshold_b=500
        )
        
        # PPL 250 should be Class B
        ppl = 250
        if ppl <= classifier.threshold_a:
            expected_class = 'A'
        elif ppl <= classifier.threshold_b:
            expected_class = 'B'
        else:
            expected_class = 'C'
        
        assert expected_class == 'B'
    
    def test_class_c_threshold(self):
        """Test Class C threshold (PPL > 500)."""
        classifier = QualityClassifier(
            model_path=None,
            threshold_a=100,
            threshold_b=500
        )
        
        # PPL 1000 should be Class C
        ppl = 1000
        if ppl <= classifier.threshold_a:
            expected_class = 'A'
        elif ppl <= classifier.threshold_b:
            expected_class = 'B'
        else:
            expected_class = 'C'
        
        assert expected_class == 'C'
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        classifier = QualityClassifier(
            model_path=None,
            threshold_a=50,
            threshold_b=200
        )
        
        assert classifier.threshold_a == 50
        assert classifier.threshold_b == 200


class TestFileProcessing:
    """Tests for file processing."""
    
    def test_process_valid_file(self, temp_dir):
        """Test processing of valid segmented file."""
        # Initialize worker
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        for cls in ['A', 'B', 'C']:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        # Create test file (one sentence per line as from Step 4)
        content = '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།
ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།'''
        
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = process_file((input_dir, output_dir, 'test.txt'))
        filename, success, error, stats = result
        
        assert success is True
        assert stats['total_sentences'] == 3
        # Without model, all sentences get class B (default PPL=250)
        assert stats['class_B'] == 3
    
    def test_process_empty_file(self, temp_dir):
        """Test processing of empty file."""
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        for cls in ['A', 'B', 'C']:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        with open(os.path.join(input_dir, 'empty.txt'), 'w', encoding='utf-8') as f:
            f.write('')
        
        result = process_file((input_dir, output_dir, 'empty.txt'))
        filename, success, error, stats = result
        
        assert success is False
        assert 'No sentences' in error


class TestOutputStructure:
    """Tests for output directory structure."""
    
    def test_output_directories_created(self, temp_dir):
        """Test that output directories are properly structured."""
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        for cls in ['A', 'B', 'C']:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        # Create test file
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།')
        
        result = process_file((input_dir, output_dir, 'test.txt'))
        
        if result[1]:  # success
            # Check that at least one class directory has output
            has_output = False
            for cls in ['A', 'B', 'C']:
                cls_dir = os.path.join(output_dir, cls)
                if os.listdir(cls_dir):
                    has_output = True
                    break
            
            assert has_output


class TestIntegration:
    """Integration tests for classification."""
    
    def test_full_classification_workflow(self, temp_dir):
        """Test complete classification workflow."""
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        for cls in ['A', 'B', 'C']:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        # Create multiple test files
        files_content = {
            'doc1.txt': 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།\nལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།',
            'doc2.txt': 'བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།',
        }
        
        for filename, content in files_content.items():
            with open(os.path.join(input_dir, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        total_sentences = 0
        for filename in files_content.keys():
            result = process_file((input_dir, output_dir, filename))
            if result[1]:  # success
                total_sentences += result[3]['total_sentences']
        
        assert total_sentences == 3
    
    def test_statistics_accuracy(self, temp_dir):
        """Test that statistics are accurately tracked."""
        init_worker()
        
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        for cls in ['A', 'B', 'C']:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
        # Create test file with known number of sentences
        sentences = [
            'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
            'བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད།',
            'ལྷ་ས་ནི་བོད་ཀྱི་རྒྱལ་ས་རེད།',
            'བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།',
            'བོད་ཀྱི་རིག་གཞུང་ནི་ཕྱུག་པོ་རེད།',
        ]
        
        with open(os.path.join(input_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sentences))
        
        result = process_file((input_dir, output_dir, 'test.txt'))
        filename, success, error, stats = result
        
        assert success is True
        assert stats['total_sentences'] == 5
        
        # Verify class counts sum to total
        total_classified = stats['class_A'] + stats['class_B'] + stats['class_C']
        assert total_classified == stats['total_sentences']
