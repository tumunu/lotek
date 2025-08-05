import argparse
import logging
from src.cybersecurity.hacker_forum_scraper import scrape_hacker_forum

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="CLI for LoTek Hacker Forum Scraper")
    parser.add_argument('--url', required=True, help='URL of the hacker forum to scrape')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='Output format (text or json). Default is text.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.debug(f'Scraping URL: {args.url}')

    try:
        discussions = scrape_hacker_forum(args.url)
        logging.debug(f'Found {len(discussions)} discussions')

        if args.format == 'json':
            import json
            print(json.dumps(discussions))
        else:
            for title in discussions:
                print(title)

    except Exception as e:
        logging.error(f'An error occurred: {e}')


if __name__ == '__main__':
    main()

