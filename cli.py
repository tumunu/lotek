#!/usr/bin/env python3
"""
LoTek CLI Wrapper
Command-line interface for LoTek scraper functionality
"""

import argparse
import json
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from cybersecurity.network_monitoring import monitor_network
    from config_management import ConfigManager, Environment
except ImportError as e:
    print(f"Error importing LoTek modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class LoTekCLI:
    """Command-line interface for LoTek scraper operations."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(Environment.PRODUCTION)
    
    def setup_logging(self, verbosity: int) -> None:
        """Configure logging based on verbosity level."""
        if verbosity >= 2:
            level = logging.DEBUG
        elif verbosity == 1:
            level = logging.INFO
        else:
            level = logging.WARNING
            
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def scrape_url(self, url: str, output_format: str, verbosity: int) -> Dict[str, Any]:
        """
        Scrape/monitor a single URL using LoTek's network monitoring.
        
        Args:
            url: Target URL to scrape/monitor
            output_format: Output format (json, text, csv)
            verbosity: Verbosity level (0-2)
            
        Returns:
            Dictionary containing scraping results
        """
        self.setup_logging(verbosity)
        logger = logging.getLogger(__name__)
        
        if verbosity >= 1:
            logger.info(f"Starting scrape operation for: {url}")
            logger.info(f"Output format: {output_format}")
        
        try:
            # Use LoTek's network monitoring as the "scraper"
            results = monitor_network([url])
            
            if verbosity >= 1:
                logger.info("Scrape operation completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return {"error": str(e), "url": url}
    
    def format_output(self, data: Dict[str, Any], output_format: str) -> str:
        """Format output according to specified format."""
        if output_format.lower() == 'json':
            return json.dumps(data, indent=2)
        
        elif output_format.lower() == 'csv':
            # Simple CSV format for network monitoring results
            lines = ["url,status,ping_success,dns_resolved"]
            for url, info in data.items():
                if isinstance(info, dict):
                    status = "ONLINE" if info.get('ping', False) else "OFFLINE"
                    ping = info.get('ping', False)
                    dns = info.get('dns_resolved', False)
                    lines.append(f"{url},{status},{ping},{dns}")
            return "\n".join(lines)
        
        elif output_format.lower() == 'text':
            # Human-readable text format
            lines = []
            for url, info in data.items():
                if isinstance(info, dict):
                    status = "ONLINE" if info.get('ping', False) else "OFFLINE"
                    lines.append(f"URL: {url}")
                    lines.append(f"Status: {status}")
                    lines.append(f"Ping Success: {info.get('ping', False)}")
                    lines.append(f"DNS Resolved: {info.get('dns_resolved', False)}")
                    lines.append("-" * 40)
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="LoTek CLI - Fractal Cybersecurity Network Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -u google.com
  %(prog)s -u github.com -f json -v
  %(prog)s -u example.com -f csv -vv
        """
    )
    
    parser.add_argument(
        '-u', '--url',
        required=True,
        help='Target URL to scrape/monitor'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'text', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (-v for INFO, -vv for DEBUG)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='LoTek CLI v1.0.0'
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize CLI
    cli = LoTekCLI()
    
    try:
        # Perform scraping operation
        results = cli.scrape_url(args.url, args.format, args.verbose)
        
        # Format output
        formatted_output = cli.format_output(results, args.format)
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            if args.verbose >= 1:
                print(f"Results written to: {args.output}")
        else:
            print(formatted_output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
