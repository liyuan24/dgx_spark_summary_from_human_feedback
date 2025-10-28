import pytest
import sys
import os
from unittest.mock import Mock
from transformers import AutoTokenizer

from dgx_spark_summary_from_human_feedback.process_dataset import (
    DatasetPreprocessingParams,
    process_query,
)


class TestProcessQuery:
    """Test cases for the process_query function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3]  # Mock encoding

        # Default test parameters
        self.params = DatasetPreprocessingParams(
            max_sft_response_length=100,
            query_length=50,
            query_format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            query_truncation_field="post",
            query_truncation_text="\n",
            query_padding_token="[PAD]",
            query_padding_side="right",
        )

        # Default test query info
        self.query_info = {
            "subreddit": "test_subreddit",
            "title": "Test Title",
            "post": "This is a test post with some content.\n\nMore content here.",
        }

    def test_process_query_basic_functionality(self):
        """Test basic functionality of process_query."""
        result = process_query(self.query_info, self.params, self.tokenizer)

        # Should return a list of integers (token IDs)
        assert isinstance(result, list)
        assert all(isinstance(token, int) for token in result)

        # Should have the correct length (padded to query_length)
        assert len(result) == self.params.query_length

    def test_process_query_with_short_content(self):
        """Test process_query with content shorter than query_length."""
        # Mock tokenizer to return short token sequence
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        result = process_query(self.query_info, self.params, self.tokenizer)

        # Should be padded to query_length
        assert len(result) == self.params.query_length
        # Should contain the original tokens plus padding
        assert result[:5] == [1, 2, 3, 4, 5]

    def test_process_query_with_long_content_truncation(self):
        """Test process_query with content that needs truncation."""

        # Mock tokenizer to return long token sequence initially
        def mock_encode(text):
            if "More content here" in text:
                return list(range(60))  # Longer than query_length
            else:
                return list(range(30))  # Shorter after truncation

        self.tokenizer.encode.side_effect = mock_encode

        result = process_query(self.query_info, self.params, self.tokenizer)

        # Should be padded to query_length
        assert len(result) == self.params.query_length

    def test_process_query_left_padding(self):
        """Test process_query with left padding."""
        self.params.query_padding_side = "left"
        self.tokenizer.encode.return_value = [1, 2, 3]

        result = process_query(self.query_info, self.params, self.tokenizer)

        # Should be padded to query_length
        assert len(result) == self.params.query_length
        # Padding should be at the beginning
        padding_token_id = self.tokenizer.encode(self.params.query_padding_token)[0]
        print(f"padding token id: {padding_token_id}")
        expected_padding_length = self.params.query_length - 3
        assert (
            result[:expected_padding_length]
            == [padding_token_id] * expected_padding_length
        )
        assert result[expected_padding_length:] == [1, 2, 3]

    def test_process_query_right_padding(self):
        """Test process_query with right padding."""
        self.params.query_padding_side = "right"
        self.tokenizer.encode.return_value = [1, 2, 3]

        result = process_query(self.query_info, self.params, self.tokenizer)

        # Should be padded to query_length
        assert len(result) == self.params.query_length
        # Padding should be at the end
        padding_token_id = self.tokenizer.encode(self.params.query_padding_token)[0]
        expected_padding_length = self.params.query_length - 3
        assert result[:3] == [1, 2, 3]
        assert result[3:] == [padding_token_id] * expected_padding_length

    def test_process_query_invalid_padding_side(self):
        """Test process_query with invalid padding side."""
        self.params.query_padding_side = "invalid"
        self.tokenizer.encode.return_value = [1, 2, 3]

        with pytest.raises(AssertionError, match="Invalid padding side: invalid"):
            process_query(self.query_info, self.params, self.tokenizer)


if __name__ == "__main__":
    pytest.main([__file__])
