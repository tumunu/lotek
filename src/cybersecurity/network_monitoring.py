"""
Cybersecurity Network Monitoring Module

Provides secure network monitoring with DNS resolution, connectivity analysis,
and input validation to prevent injection attacks.
"""

import subprocess
import socket
import re
import ipaddress
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_hostname(host: str) -> bool:
    """Validate hostname/IP to prevent command injection."""
    if not isinstance(host, str) or len(host) > 253:
        return False
    
    # Try as IP address first
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    
    # Validate as hostname (RFC 1123)
    hostname_pattern = re.compile(
        r'^(?=.{1,253}$)(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)*(?!-)[A-Za-z0-9-]{1,63}(?<!-)$'
    )
    return bool(hostname_pattern.match(host))

def ping(host: str) -> bool:
    """Ping a host and return True if reachable, False otherwise."""
    if not validate_hostname(host):
        logger.warning(f"Invalid hostname format: {host}")
        return False
    
    try:
        # Use timeout and limit packet count for security
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '3', host],
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )
        return result.returncode == 0 and '1 received' in result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.error(f"Ping failed for {host}: {e}")
        return False

def nslookup(host: str) -> Optional[str]:
    """Perform DNS lookup to get IP address for a host."""
    if not validate_hostname(host):
        logger.warning(f"Invalid hostname format for DNS lookup: {host}")
        return None
    
    try:
        # Set timeout for DNS resolution
        socket.setdefaulttimeout(5.0)
        ip = socket.gethostbyname(host)
        logger.info(f"DNS lookup successful for {host}: {ip}")
        return ip
    except (socket.gaierror, socket.timeout) as e:
        logger.error(f"DNS lookup failed for {host}: {e}")
        return None
    finally:
        socket.setdefaulttimeout(None)

def monitor_network(hosts: List[str], max_concurrent: int = 10) -> Dict[str, Dict[str, Union[bool, str, None]]]:
    """Monitor network status for a list of hosts with rate limiting."""
    if not isinstance(hosts, list) or len(hosts) > max_concurrent:
        raise ValueError(f"Invalid hosts list or too many hosts (max: {max_concurrent})")
    
    network_status = defaultdict(dict)
    valid_hosts = [host for host in hosts if validate_hostname(host)]
    
    if len(valid_hosts) != len(hosts):
        logger.warning(f"Filtered {len(hosts) - len(valid_hosts)} invalid hosts")
    
    for host in valid_hosts:
        logger.info(f"Monitoring host: {host}")
        network_status[host]['ping'] = ping(host)
        network_status[host]['ip'] = nslookup(host)
        network_status[host]['timestamp'] = __import__('time').time()
    
    return dict(network_status)