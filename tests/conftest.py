"""
Pytest configuration and shared fixtures for pipeline tests.
"""

import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

# Sample Tibetan text fixtures
TIBETAN_SAMPLES = {
    'clean_sentence': 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
    'clean_paragraph': '''བོད་ནི་ཨེ་ཤི་ཡའི་དབུས་སུ་ཡོད་པའི་ས་ཁུལ་ཞིག་རེད།
བོད་ཀྱི་རྒྱལ་ས་ནི་ལྷ་ས་རེད།
བོད་མི་རྣམས་བོད་སྐད་བཤད་ཀྱི་ཡོད།''',
    'mixed_content': '''བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།
This is English text mixed in.
བོད་ཀྱི་རྒྱལ་ས་ནི་ལྷ་ས་རེད།
More English content here.''',
    'short_sentence': 'བོད།',
    'noisy_text': 'བོད asdfjkl; @#$%^ random123 སྐད།',
    'pure_english': 'This is purely English text with no Tibetan characters.',
    'duplicate_text': 'བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད། བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་རེད།',
}


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_tibetan_file(temp_dir):
    """Create a sample Tibetan text file."""
    filepath = os.path.join(temp_dir, 'sample.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(TIBETAN_SAMPLES['clean_paragraph'])
    return filepath


@pytest.fixture
def sample_mixed_file(temp_dir):
    """Create a mixed content file."""
    filepath = os.path.join(temp_dir, 'mixed.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(TIBETAN_SAMPLES['mixed_content'])
    return filepath


@pytest.fixture
def sample_english_file(temp_dir):
    """Create a pure English file."""
    filepath = os.path.join(temp_dir, 'english.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(TIBETAN_SAMPLES['pure_english'])
    return filepath


@pytest.fixture
def input_dir_with_files(temp_dir):
    """Create input directory with multiple test files."""
    input_dir = os.path.join(temp_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    
    # Create various test files
    files = {
        'clean1.txt': TIBETAN_SAMPLES['clean_paragraph'],
        'clean2.txt': TIBETAN_SAMPLES['clean_sentence'] * 5,
        'mixed.txt': TIBETAN_SAMPLES['mixed_content'],
        'short.txt': TIBETAN_SAMPLES['short_sentence'],
        'english.txt': TIBETAN_SAMPLES['pure_english'],
    }
    
    for filename, content in files.items():
        with open(os.path.join(input_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)
    
    return input_dir


@pytest.fixture
def output_dir(temp_dir):
    """Create output directory."""
    out_dir = os.path.join(temp_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
