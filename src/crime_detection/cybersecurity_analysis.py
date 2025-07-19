# 

from src.crime_detection.ippsec_training import ippsec_training
from src.crime_detection.htb_training import htb_training

def cybersecurity_training():
    print("Starting IppSec Training")
    ippsec_training()
    
    print("Starting Hack The Box Training")
    htb_training()

if __name__ == "__main__":
    cybersecurity_training()
