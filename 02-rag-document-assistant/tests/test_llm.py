
from unittest.mock import MagicMock, patch

from src.llm import generate_answer


@patch("src.llm.client.chat.completions.create")
def test_generate_answer(mock_create):
    # Arrange: Prepare a fake LLM response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "Artificial Intelligence is technology "
        "that enables machines to perform intelligent tasks."
    )

    mock_create.return_value = mock_response

    # Act: Call the function
    result = generate_answer("What is AI?")

    # Assert: Check the result
    assert isinstance(result, str)
    assert result == (
        "Artificial Intelligence is technology "
        "that enables machines to perform intelligent tasks."
    )

    # Check that the API was called correctly
    mock_create.assert_called_once_with(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "What is AI?"
            }
        ],
        temperature=0.7,
        max_completion_tokens=1024
    )
