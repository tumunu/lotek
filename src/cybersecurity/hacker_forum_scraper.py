# hacker_forum_scraper.py - Cybersecurity Hacker Forum Scraper Module

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ScraperError(Exception):
    """Base exception for scraper errors"""
    pass

class NetworkError(ScraperError):
    """Network-related scraping errors"""
    pass

class ParsingError(ScraperError):
    """HTML parsing errors"""
    pass

def scrape_hacker_forum(url: str, timeout: int = 10) -> List[str]:
    """
    Scrape hacker forums for discussions on vulnerabilities or exploits.
    
    Args:
        url: Valid forum URL to scrape
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        List of discussion titles
        
    Raises:
        NetworkError: For connection/timeout issues
        ParsingError: For HTML parsing failures
        ValueError: For invalid URL format
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': 'LoTek Security Scanner'}
        )
        response.raise_for_status()
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise NetworkError("Request timed out") from None
    except requests.exceptions.SSLError:
        logger.error("SSL verification failed")
        raise NetworkError("SSL verification failed") from None
    except requests.exceptions.ConnectionError:
        logger.error("Connection failed")
        raise NetworkError("Connection failed") from None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code}")
        raise NetworkError(f"HTTP error: {e.response.status_code}") from None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise NetworkError("Request failed") from None

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = soup.find_all('div', {"class": "post"})
        discussions = []
        for post in posts:
            title = post.find('h3', {"class": "post-title"})
            if title:
                discussions.append(title.text.strip())
        return discussions
    except Exception as e:
        logger.error(f"HTML parsing failed: {str(e)}")
        raise ParsingError("Failed to parse HTML content") from None
