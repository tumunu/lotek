# test_hacker_forum_scraper.py - Comprehensive tests for hacker_forum_scraper

import pytest
from unittest.mock import patch, Mock
from src.cybersecurity.hacker_forum_scraper import (
    scrape_hacker_forum,
    NetworkError,
    ParsingError
)

@pytest.fixture
def mock_html_response():
    html = """
    <html>
        <div class="post">
            <h3 class="post-title">Test Post</h3>
        </div>
    </html>
    """
    response = Mock()
    response.text = html
    response.status_code = 200
    return response

def test_successful_scrape(mock_html_response):
    """Test successful scraping of valid HTML"""
    with patch('requests.get', return_value=mock_html_response):
        results = scrape_hacker_forum("https://valid.forum")
        assert len(results) == 1
        assert "Test Post" in results

def test_timeout_handling():
    """Test timeout error handling"""
    with patch('requests.get', side_effect=requests.exceptions.Timeout):
        with pytest.raises(NetworkError, match="Request timed out"):
            scrape_hacker_forum("https://timeout.forum")

def test_ssl_error_handling():
    """Test SSL error handling"""
    with patch('requests.get', side_effect=requests.exceptions.SSLError):
        with pytest.raises(NetworkError, match="SSL verification failed"):
            scrape_hacker_forum("https://ssl-error.forum")

def test_connection_error_handling():
    """Test connection error handling"""
    with patch('requests.get', side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(NetworkError, match="Connection failed"):
            scrape_hacker_forum("https://connection-error.forum")

def test_http_error_handling():
    """Test HTTP error handling"""
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "403 Forbidden", response=mock_response
    )
    with patch('requests.get', return_value=mock_response):
        with pytest.raises(NetworkError, match="HTTP error: 403"):
            scrape_hacker_forum("https://forbidden.forum")

def test_html_parsing_error():
    """Test HTML parsing error handling"""
    mock_response = Mock()
    mock_response.text = "Invalid HTML"
    mock_response.status_code = 200
    with patch('requests.get', return_value=mock_response):
        with pytest.raises(ParsingError, match="Failed to parse HTML content"):
            scrape_hacker_forum("https://invalid-html.forum")

def test_empty_response_handling():
    """Test empty response handling"""
    mock_response = Mock()
    mock_response.text = ""
    mock_response.status_code = 200
    with patch('requests.get', return_value=mock_response):
        results = scrape_hacker_forum("https://empty.forum")
        assert results == []
