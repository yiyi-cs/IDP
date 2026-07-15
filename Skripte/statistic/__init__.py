"""
=================================================================================
STATISTIC PACKAGE - Statistische Analyse fuer Eye-Tracking-Pipeline
=================================================================================

Module:
-------
- data_availability_scanner: Prueft vorhandene Analysedaten
- statistical_data_prep: Erstellt einheitlichen Analysedatensatz
- statistical_pipeline: Orchestriert statistische Analyse
- statistical_cli: Interaktives Terminal
- jitter_filter: Einheitliche Jitter-Reduktion

Version: 1.0
Datum: 2025-01
=================================================================================
"""

from .data_availability_scanner import DataAvailabilityScanner, DataAvailabilityReport
from .statistical_data_prep import StatisticalDataPrep
from .statistical_pipeline import StatisticalPipeline

__version__ = '1.0'
__all__ = [
    'DataAvailabilityScanner',
    'DataAvailabilityReport', 
    'StatisticalDataPrep',
    'StatisticalPipeline'
]
