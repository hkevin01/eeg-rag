"""
Tests for EEG Named Entity Recognition (NER) System
"""

import pytest
from src.eeg_rag.nlp.ner_eeg import (
    EEGNER, Entity, EntityType, NERResult, EEGTerminologyDatabase
)


class TestEEGTerminologyDatabase:
    """Test the EEG terminology database"""
    
    def test_frequency_bands_defined(self):
        """Test that frequency bands are properly defined"""
        db = EEGTerminologyDatabase()
        
        assert 'delta' in db.FREQUENCY_BANDS
        assert 'theta' in db.FREQUENCY_BANDS
        assert 'alpha' in db.FREQUENCY_BANDS
        assert 'beta' in db.FREQUENCY_BANDS
        assert 'gamma' in db.FREQUENCY_BANDS
        
        # Check frequency ranges
        assert db.FREQUENCY_BANDS['alpha']['range'] == (8, 13)
        assert db.FREQUENCY_BANDS['theta']['range'] == (4, 8)
    
    def test_brain_regions_defined(self):
        """Test that brain regions are defined"""
        db = EEGTerminologyDatabase()
        
        assert 'frontal cortex' in db.BRAIN_REGIONS
        assert 'temporal lobe' in db.BRAIN_REGIONS
        assert 'hippocampus' in db.BRAIN_REGIONS
        assert len(db.BRAIN_REGIONS) > 30
    
    def test_electrodes_defined(self):
        """Test that electrode locations are defined"""
        db = EEGTerminologyDatabase()
        
        assert 'Fp1' in db.ELECTRODES
        assert 'Cz' in db.ELECTRODES
        assert 'O1' in db.ELECTRODES
        assert len(db.ELECTRODES) > 50
    
    def test_clinical_conditions_defined(self):
        """Test that clinical conditions are defined"""
        db = EEGTerminologyDatabase()
        
        assert 'epilepsy' in db.CLINICAL_CONDITIONS
        assert 'depression' in db.CLINICAL_CONDITIONS
        assert "alzheimer's disease" in db.CLINICAL_CONDITIONS
        assert len(db.CLINICAL_CONDITIONS) > 40
    
    def test_biomarkers_defined(self):
        """Test that EEG biomarkers are defined"""
        db = EEGTerminologyDatabase()
        
        assert 'P300' in db.BIOMARKERS
        assert 'alpha asymmetry' in db.BIOMARKERS
        assert 'theta-beta ratio' in db.BIOMARKERS
        assert len(db.BIOMARKERS) > 30


class TestEEGNER:
    """Test the EEG NER system"""
    
    def test_initialization(self):
        """Test NER system initialization"""
        ner = EEGNER()
        
        assert ner.terminology is not None
        assert len(ner.compiled_patterns) > 0
        assert ner.stats['documents_processed'] == 0
    
    def test_extract_frequency_bands(self):
        """Test extraction of frequency band entities"""
        ner = EEGNER()
        
        text = "We observed increased theta and alpha activity in the frontal regions."
        result = ner.extract_entities(text)
        
        # Check that entities were found
        assert len(result.entities) > 0
        
        # Check for frequency bands
        band_entities = [e for e in result.entities if e.entity_type == EntityType.FREQUENCY_BAND]
        assert len(band_entities) >= 2
        
        # Check metadata for frequency bands
        theta_entity = next((e for e in band_entities if e.text.lower() == 'theta'), None)
        if theta_entity:
            assert 'frequency_range' in theta_entity.metadata
            assert theta_entity.metadata['frequency_range'] == (4, 8)
    
    def test_extract_brain_regions(self):
        """Test extraction of brain region entities"""
        ner = EEGNER()
        
        text = "Activity in the frontal cortex and hippocampus was recorded."
        result = ner.extract_entities(text)
        
        brain_regions = [e for e in result.entities if e.entity_type == EntityType.BRAIN_REGION]
        assert len(brain_regions) >= 2
        
        region_texts = [e.text.lower() for e in brain_regions]
        assert any('frontal' in text for text in region_texts)
        assert any('hippocampus' in text for text in region_texts)
    
    def test_extract_electrodes(self):
        """Test extraction of electrode entities"""
        ner = EEGNER()
        
        text = "Electrodes Fp1, Cz, and O1 were used for recording."
        result = ner.extract_entities(text)
        
        electrodes = [e for e in result.entities if e.entity_type == EntityType.ELECTRODE]
        assert len(electrodes) >= 3
        
        electrode_names = [e.text for e in electrodes]
        assert 'Fp1' in electrode_names
        assert 'Cz' in electrode_names
        assert 'O1' in electrode_names
    
    def test_extract_clinical_conditions(self):
        """Test extraction of clinical condition entities"""
        ner = EEGNER()
        
        text = "Patients with epilepsy and depression were included in the study."
        result = ner.extract_entities(text)
        
        conditions = [e for e in result.entities if e.entity_type == EntityType.CLINICAL_CONDITION]
        assert len(conditions) >= 2
        
        condition_texts = [e.text.lower() for e in conditions]
        assert 'epilepsy' in condition_texts
        assert 'depression' in condition_texts
    
    def test_extract_biomarkers(self):
        """Test extraction of biomarker entities"""
        ner = EEGNER()
        
        text = "The P300 amplitude and alpha asymmetry were analyzed."
        result = ner.extract_entities(text)
        
        biomarkers = [e for e in result.entities if e.entity_type == EntityType.BIOMARKER]
        assert len(biomarkers) >= 2
        
        biomarker_texts = [e.text for e in biomarkers]
        assert 'P300' in biomarker_texts or 'amplitude' in biomarker_texts
    
    def test_extract_multiple_entity_types(self):
        """Test extraction of multiple entity types from complex text"""
        ner = EEGNER()
        
        text = """
        EEG recordings were obtained from electrodes Fp1, Fz, and Cz during an 
        oddball task. Increased theta power was observed in the frontal cortex 
        of patients with epilepsy. The P300 amplitude showed significant differences 
        between groups.
        """
        
        result = ner.extract_entities(text)
        
        # Should find multiple entity types
        assert len(result.entity_counts) >= 4
        
        # Check specific types are present
        entity_types = [e.entity_type.value for e in result.entities]
        assert 'electrode' in entity_types
        assert 'frequency_band' in entity_types
        assert 'brain_region' in entity_types
        assert 'clinical_condition' in entity_types
    
    def test_confidence_scoring(self):
        """Test that confidence scores are calculated"""
        ner = EEGNER()
        
        text = "Alpha waves were measured in the frontal cortex."
        result = ner.extract_entities(text)
        
        # All entities should have confidence scores
        for entity in result.entities:
            assert 0.0 <= entity.confidence <= 1.0
    
    def test_context_extraction(self):
        """Test that context is extracted around entities"""
        ner = EEGNER()
        
        text = "The study examined alpha activity in resting state conditions."
        result = ner.extract_entities(text, context_window=20)
        
        # Entities should have context
        for entity in result.entities:
            assert len(entity.context) > len(entity.text)
            assert entity.text in entity.context
    
    def test_overlap_removal(self):
        """Test that overlapping entities are handled correctly"""
        ner = EEGNER()
        
        # Text with potentially overlapping terms
        text = "Frontal lobe activity in the frontal cortex was analyzed."
        result = ner.extract_entities(text)
        
        # Check that no entities overlap
        sorted_entities = sorted(result.entities, key=lambda e: e.start_pos)
        for i in range(len(sorted_entities) - 1):
            assert sorted_entities[i].end_pos <= sorted_entities[i+1].start_pos
    
    def test_batch_processing(self):
        """Test batch processing of multiple texts"""
        ner = EEGNER()
        
        texts = [
            "Alpha and theta waves were recorded.",
            "Patients with epilepsy showed abnormal activity.",
            "The P300 amplitude was measured at electrode Cz."
        ]
        
        results = ner.extract_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, NERResult) for r in results)
        assert all(len(r.entities) > 0 for r in results)
    
    def test_entity_summary(self):
        """Test entity summary generation"""
        ner = EEGNER()
        
        text = "Alpha, beta, and gamma waves were analyzed in epilepsy patients."
        result = ner.extract_entities(text)
        
        summary = ner.get_entity_summary(result)
        
        assert 'total_entities' in summary
        assert 'entity_types' in summary
        assert 'entity_counts' in summary
        assert 'unique_entities' in summary
        assert 'avg_confidence' in summary
        
        assert summary['total_entities'] > 0
        assert summary['avg_confidence'] > 0
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly"""
        ner = EEGNER()
        
        texts = [
            "Alpha waves in frontal cortex.",
            "Beta activity at electrode Cz.",
            "Theta oscillations in epilepsy."
        ]
        
        for text in texts:
            ner.extract_entities(text)
        
        stats = ner.get_statistics()
        
        assert stats['documents_processed'] == 3
        assert stats['total_entities_found'] > 0
        assert stats['avg_entities_per_doc'] > 0
        assert len(stats['entities_by_type']) > 0
    
    def test_minimum_confidence_filtering(self):
        """Test filtering entities by minimum confidence"""
        ner = EEGNER()
        
        text = "Alpha waves in the brain during sleep."
        
        # Extract with low threshold
        result_low = ner.extract_entities(text, min_confidence=0.0)
        
        # Extract with high threshold
        result_high = ner.extract_entities(text, min_confidence=0.95)
        
        # High confidence should have fewer or equal entities
        assert len(result_high.entities) <= len(result_low.entities)
    
    def test_processing_time_recorded(self):
        """Test that processing time is recorded"""
        ner = EEGNER()
        
        text = "EEG analysis of alpha, beta, and gamma waves."
        result = ner.extract_entities(text)
        
        assert result.processing_time > 0
        assert result.processing_time < 1.0  # Should be fast
    
    def test_entity_to_dict(self):
        """Test entity serialization to dictionary"""
        ner = EEGNER()
        
        text = "Alpha waves at 10 Hz."
        result = ner.extract_entities(text)
        
        if result.entities:
            entity_dict = result.entities[0].to_dict()
            
            assert 'text' in entity_dict
            assert 'type' in entity_dict
            assert 'start' in entity_dict
            assert 'end' in entity_dict
            assert 'confidence' in entity_dict
    
    def test_result_to_dict(self):
        """Test NER result serialization to dictionary"""
        ner = EEGNER()
        
        text = "Theta and alpha activity."
        result = ner.extract_entities(text)
        
        result_dict = result.to_dict()
        
        assert 'text_length' in result_dict
        assert 'total_entities' in result_dict
        assert 'entities' in result_dict
        assert 'entity_counts' in result_dict
        assert 'processing_time' in result_dict
    
    def test_case_insensitive_matching(self):
        """Test that entity matching is case-insensitive"""
        ner = EEGNER()
        
        text1 = "ALPHA waves were detected."
        text2 = "alpha waves were detected."
        text3 = "Alpha waves were detected."
        
        result1 = ner.extract_entities(text1)
        result2 = ner.extract_entities(text2)
        result3 = ner.extract_entities(text3)
        
        # All should detect alpha
        assert len(result1.entities) > 0
        assert len(result2.entities) > 0
        assert len(result3.entities) > 0


class TestRealWorldScenarios:
    """Test NER on realistic research paper excerpts"""
    
    def test_research_abstract(self):
        """Test NER on a realistic research abstract"""
        ner = EEGNER()
        
        abstract = """
        Background: This study investigates theta and alpha oscillations in patients 
        with epilepsy during resting state. Methods: EEG was recorded from 32 electrodes 
        including Fp1, Fz, Cz, and O1 using a high-density EEG system. Independent 
        component analysis (ICA) was used for artifact removal. Results: Patients showed 
        increased theta power in the frontal cortex compared to controls (p < 0.001). 
        Alpha asymmetry was significantly correlated with seizure frequency. The P300 
        amplitude during an oddball task was reduced in patients. Conclusion: These 
        findings suggest that quantitative EEG measures provide useful biomarkers for 
        epilepsy diagnosis.
        """
        
        result = ner.extract_entities(abstract)
        
        # Should find many entities
        assert len(result.entities) >= 10
        
        # Should include multiple entity types
        assert len(result.entity_counts) >= 5
        
        # Check for key terms
        entity_texts = [e.text.lower() for e in result.entities]
        assert any('theta' in text for text in entity_texts)
        assert any('alpha' in text for text in entity_texts)
        assert any('epilepsy' in text for text in entity_texts)
        
        # Get summary
        summary = ner.get_entity_summary(result)
        assert summary['total_entities'] >= 10
    
    def test_methods_section(self):
        """Test NER on a methods section"""
        ner = EEGNER()
        
        methods = """
        EEG data were collected using a 64-channel system with electrodes placed 
        according to the 10-20 international system. Signals were sampled at 1000 Hz 
        and bandpass filtered between 0.5 and 100 Hz. A notch filter at 60 Hz was 
        applied to remove power line noise. Independent component analysis (ICA) was 
        used to identify and remove ocular artifacts. Data were then segmented into 
        2-second epochs and analyzed using fast Fourier transform (FFT) to extract 
        power spectral density (PSD) for delta, theta, alpha, beta, and gamma bands.
        """
        
        result = ner.extract_entities(methods)
        
        # Should find processing methods
        methods_entities = [e for e in result.entities if e.entity_type == EntityType.PROCESSING_METHOD]
        assert len(methods_entities) >= 3
        
        # Should find frequency bands
        band_entities = [e for e in result.entities if e.entity_type == EntityType.FREQUENCY_BAND]
        assert len(band_entities) >= 4
        
        # Should find measurement units
        unit_entities = [e for e in result.entities if e.entity_type == EntityType.MEASUREMENT_UNIT]
        assert len(unit_entities) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
