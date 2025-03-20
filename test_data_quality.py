#!/usr/bin/env python3
"""
Unit tests for the data quality framework.

This module contains tests for the core components of the data quality framework,
including rule evaluation, validation, and reporting.
"""

import unittest
import pandas as pd
import numpy as np
import os
import json
import tempfile
from datetime import datetime

# Import the data quality framework
from data_quality import (
    DataQualityRule, CompletenessRule, UniquenessRule, ValueRangeRule,
    PatternMatchRule, StatisticalRule, CustomPythonRule, 
    DataQualityValidation, DataQualityFramework,
    create_completeness_rule, create_uniqueness_rule, create_value_range_rule,
    create_pattern_match_rule, create_statistical_rule, create_custom_python_rule
)


class TestCompletenessRule(unittest.TestCase):
    """Test the completeness rule implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['John', 'Jane', 'Bob', None, 'Alice', 'John', 'Eve', 'Charlie', 'Diana', 'Frank'],
            'age': [25, 30, 45, 27, 22, 25, None, 33, 41, 38]
        })
    
    def test_perfect_completeness(self):
        """Test a column with no nulls."""
        rule = CompletenessRule('test_id_complete', ['id'], threshold=1.0)
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        self.assertEqual(result['metrics']['columns_failed'], 0)
    
    def test_partial_completeness(self):
        """Test a column with some nulls."""
        rule = CompletenessRule('test_name_complete', ['name'], threshold=0.9)
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        self.assertEqual(result['metrics']['columns_failed'], 0)
        
        # Test with higher threshold (should fail)
        rule = CompletenessRule('test_name_complete_high', ['name'], threshold=1.0)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['columns_failed'], 1)
    
    def test_multiple_columns(self):
        """Test multiple columns at once."""
        rule = CompletenessRule('test_multiple_complete', ['name', 'age'], threshold=0.8)
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        
        # Check detailed metrics
        self.assertEqual(result['metrics']['columns_checked'], 2)
        self.assertAlmostEqual(result['metrics']['completion_rates']['name'], 0.9)
        self.assertAlmostEqual(result['metrics']['completion_rates']['age'], 0.9)


class TestUniquenessRule(unittest.TestCase):
    """Test the uniqueness rule implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['John', 'Jane', 'Bob', 'Dave', 'Alice', 'John', 'Eve', 'Charlie', 'Diana', 'Frank'],
            'category': ['A', 'B', 'A', 'B', 'C', 'A', 'C', 'B', 'A', 'C']
        })
    
    def test_perfect_uniqueness(self):
        """Test a column with all unique values."""
        rule = UniquenessRule('test_id_unique', ['id'], threshold=1.0)
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        self.assertEqual(result['metrics']['duplicate_count'], 0)
    
    def test_partial_uniqueness(self):
        """Test a column with some duplicates."""
        rule = UniquenessRule('test_name_unique', ['name'], threshold=0.9)
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        self.assertEqual(result['metrics']['duplicate_count'], 1)
        
        # Test with higher threshold (should fail)
        rule = UniquenessRule('test_name_unique_high', ['name'], threshold=1.0)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
    
    def test_composite_uniqueness(self):
        """Test uniqueness across multiple columns."""
        # 'category' alone has many duplicates
        rule = UniquenessRule('test_category_unique', ['category'], threshold=1.0)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        
        # But 'name' + 'category' should be unique
        rule = UniquenessRule('test_composite_unique', ['name', 'category'], threshold=1.0)
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])


class TestValueRangeRule(unittest.TestCase):
    """Test the value range rule implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'age': [25, 30, 45, 17, 22, 65, 28, 33, 41, 38],
            'score': [85, 92, 78, 95, 88, 75, 91, 83, 79, 90],
            'price': [-10, 10.5, 25.99, 0, 15.75, 30, 8.25, 12, 9.99, 18.50]
        })
    
    def test_inclusive_range(self):
        """Test inclusive range checks."""
        rule = ValueRangeRule('test_age_range', 'age', min_value=18, max_value=65)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['out_of_range_count'], 1)
        
        # Check details
        self.assertEqual(result['details']['min_value'], 18)
        self.assertEqual(result['details']['max_value'], 65)
    
    def test_exclusive_range(self):
        """Test exclusive range checks."""
        rule = ValueRangeRule('test_score_range', 'score', 
                            min_value=75, max_value=95,
                            include_min=False, include_max=False)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['out_of_range_count'], 2)
    
    def test_partial_range(self):
        """Test range with only min or max."""
        # Only min value
        rule = ValueRangeRule('test_price_min', 'price', min_value=0)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['out_of_range_count'], 1)
        
        # Only max value
        rule = ValueRangeRule('test_price_max', 'price', max_value=20)
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['out_of_range_count'], 2)


class TestPatternMatchRule(unittest.TestCase):
    """Test the pattern match rule implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'email': [
                'john@example.com',
                'jane@example.com',
                'bob@test.com',
                'invalid-email',
                'alice@example.com',
                'john.doe@example.com',
                'eve@example',         # Missing TLD
                'charlie@test.com',
                'diana@example.com',
                'frank@test.net'
            ],
            'phone': [
                '123-456-7890',
                '(123) 456-7890',
                '123.456.7890',
                '12345',              # Too short
                '123-456-78901',      # Too long
                '123-4567890',
                'abc-def-ghij',       # Invalid format
                '123 456 7890',
                '123456789',          # Too short
                '+1 123-456-7890'
            ]
        })
    
    def test_email_pattern(self):
        """Test email validation pattern."""
        rule = PatternMatchRule(
            'test_email_pattern',
            'email',
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            match_type='fullmatch',
            threshold=0.9
        )
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['non_matching_count'], 2)
        
        # Try with lower threshold
        rule = PatternMatchRule(
            'test_email_pattern_lower',
            'email',
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            match_type='fullmatch',
            threshold=0.8
        )
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
    
    def test_phone_pattern(self):
        """Test phone number validation pattern."""
        rule = PatternMatchRule(
            'test_phone_pattern',
            'phone',
            pattern=r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4}$',
            match_type='fullmatch',
            threshold=0.8
        )
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        
        # Check non-matching examples
        self.assertGreater(len(result['details']['non_matching_examples']), 0)


class TestStatisticalRule(unittest.TestCase):
    """Test the statistical rule implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'value': [10, 12, 15, 8, 9, 11, 14, 10, 9, 12]
        })
    
    def test_mean_check(self):
        """Test mean value check."""
        # The mean of the values is 11
        rule = StatisticalRule(
            'test_mean', 'value', 'mean', 
            expected_value=11, tolerance=0.5
        )
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        
        # Test with tighter tolerance (should fail)
        rule = StatisticalRule(
            'test_mean_tight', 'value', 'mean', 
            expected_value=11, tolerance=0.1
        )
        result = rule.evaluate(self.data)
        self.assertFalse(result['success'])
        
        # Check details
        self.assertAlmostEqual(result['metrics']['actual_value'], 11.0, places=2)
    
    def test_median_check(self):
        """Test median value check."""
        # The median of the values is 10.5
        rule = StatisticalRule(
            'test_median', 'value', 'median', 
            expected_value=10.5, tolerance=0.5
        )
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
    
    def test_std_check(self):
        """Test standard deviation check."""
        # The std of the values is about 2.3
        rule = StatisticalRule(
            'test_std', 'value', 'std', 
            expected_value=2.3, tolerance=0.5
        )
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        
        # Check that metrics include deviation information
        self.assertIn('deviation', result['metrics'])
        self.assertIn('deviation_percentage', result['metrics'])


class TestCustomPythonRule(unittest.TestCase):
    """Test the custom Python rule implementation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'start_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']),
            'end_date': pd.to_datetime(['2023-01-31', '2023-02-28', '2023-02-15', '2023-05-01'])
        })
    
    def test_custom_function(self):
        """Test a custom validation function."""
        
        def check_date_ranges(df, **kwargs):
            """Check that end dates are after start dates."""
            invalid_count = (df['end_date'] <= df['start_date']).sum()
            success = invalid_count == 0
            return {
                'success': success,
                'message': f"Found {invalid_count} invalid date ranges",
                'metrics': {'invalid_count': invalid_count}
            }
        
        rule = CustomPythonRule(
            'test_date_ranges', check_function=check_date_ranges,
            description="Check that end dates are after start dates"
        )
        result = rule.evaluate(self.data)
        self.assertTrue(result['success'])
        
        # Create invalid data
        invalid_data = self.data.copy()
        invalid_data.loc[2, 'end_date'] = pd.Timestamp('2023-02-15')
        invalid_data.loc[2, 'start_date'] = pd.Timestamp('2023-02-20')
        
        result = rule.evaluate(invalid_data)
        self.assertFalse(result['success'])
        self.assertEqual(result['metrics']['invalid_count'], 1)


class TestDataQualityValidation(unittest.TestCase):
    """Test the data quality validation class."""
    
    def setUp(self):
        """Set up test data and rules."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['John', 'Jane', 'Bob', None, 'Alice', 'John', 'Eve', 'Charlie', 'Diana', 'Frank'],
            'age': [25, 30, 45, 27, 22, 25, None, 33, 41, 38],
            'email': ['john@example.com', 'jane@example.com', 'bob@test.com', 'dave@example.com',
                     'alice@example.com', 'john.doe@example.com', 'eve@example.com', 'charlie@test.com',
                     'diana@example.com', 'frank@test.net']
        })
        
        # Create sample rules
        self.rules = [
            CompletenessRule('completeness_name', ['name'], threshold=0.9),
            CompletenessRule('completeness_age', ['age'], threshold=0.9),
            UniquenessRule('uniqueness_id', ['id'], threshold=1.0),
            UniquenessRule('uniqueness_name', ['name'], threshold=0.9),
            PatternMatchRule('pattern_email', 'email', 
                           pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                           match_type='fullmatch', threshold=1.0)
        ]
    
    def test_validation_execution(self):
        """Test executing validation on multiple rules."""
        validator = DataQualityValidation(self.rules)
        validation_report = validator.execute(self.data)
        
        # Check summary
        summary = validation_report['summary']
        self.assertEqual(summary['total_rules'], 5)
        self.assertEqual(summary['passed_rules'], 5)
        self.assertEqual(summary['failed_rules'], 0)
        self.assertEqual(summary['overall_status'], 'PASS')
        
        # Check results
        results = validation_report['results']
        self.assertEqual(len(results), 5)
        
        # Each result should have required keys
        for result in results:
            self.assertIn('rule_id', result)
            self.assertIn('success', result)
            self.assertIn('message', result)
            self.assertIn('metrics', result)
    
    def test_save_report(self):
        """Test saving validation report to file."""
        validator = DataQualityValidation(self.rules)
        validator.execute(self.data)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            report_path = tmp.name
        
        try:
            # Save report
            validator.save_report(report_path)
            
            # Check that file exists
            self.assertTrue(os.path.exists(report_path))
            
            # Load and verify content
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            self.assertIn('validation_id', report_data)
            self.assertIn('results', report_data)
            self.assertEqual(len(report_data['results']), 5)
            
        finally:
            # Clean up
            if os.path.exists(report_path):
                os.unlink(report_path)


class TestDataQualityFramework(unittest.TestCase):
    """Test the data quality framework class."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['John', 'Jane', 'Bob', None, 'Alice', 'John', 'Eve', 'Charlie', 'Diana', 'Frank'],
            'age': [25, 30, 45, 27, 22, 25, None, 33, 41, 38],
            'email': ['john@example.com', 'jane@example.com', 'bob@test.com', 'dave@example.com',
                     'alice@example.com', 'john.doe@example.com', 'eve@example.com', 'charlie@test.com',
                     'diana@example.com', 'frank@test.net']
        })
    
    def test_rule_management(self):
        """Test adding, getting, and removing rules."""
        framework = DataQualityFramework()
        
        # Add a rule
        rule = CompletenessRule('test_rule', ['name'])
        framework.add_rule(rule)
        
        # Get the rule
        retrieved_rule = framework.get_rule('test_rule')
        self.assertEqual(retrieved_rule.rule_id, 'test_rule')
        
        # Remove the rule
        result = framework.remove_rule('test_rule')
        self.assertTrue(result)
        
        # Rule should be gone
        self.assertIsNone(framework.get_rule('test_rule'))
    
    def test_rule_groups(self):
        """Test creating and using rule groups."""
        framework = DataQualityFramework()
        
        # Add rules
        framework.add_rule(CompletenessRule('completeness_name', ['name']))
        framework.add_rule(CompletenessRule('completeness_age', ['age']))
        framework.add_rule(UniquenessRule('uniqueness_id', ['id']))
        
        # Create a rule group
        framework.create_rule_group('completeness_checks', 
                                   ['completeness_name', 'completeness_age'])
        
        # Get rules by group
        group_rules = framework.get_rules_by_group('completeness_checks')
        self.assertEqual(len(group_rules), 2)
        
        # Rule IDs should match
        rule_ids = [rule.rule_id for rule in group_rules]
        self.assertIn('completeness_name', rule_ids)
        self.assertIn('completeness_age', rule_ids)
    
    def test_validate_data(self):
        """Test validating data with the framework."""
        framework = DataQualityFramework()
        
        # Add rules
        framework.add_rule(CompletenessRule('completeness_name', ['name'], threshold=0.9))
        framework.add_rule(UniquenessRule('uniqueness_id', ['id']))
        
        # Validate data
        report = framework.validate_data(self.data)
        
        # Check report
        self.assertEqual(report['summary']['total_rules'], 2)
        self.assertEqual(report['summary']['passed_rules'], 2)
        
        # Create a rule group and validate against it
        framework.add_rule(CompletenessRule('completeness_email', ['email']))
        framework.create_rule_group('id_checks', ['uniqueness_id'])
        
        group_report = framework.validate_data(self.data, group_name='id_checks')
        self.assertEqual(group_report['summary']['total_rules'], 1)


class TestHelperFunctions(unittest.TestCase):
    """Test the helper functions for creating rules."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['John', 'Jane', 'Bob', None, 'Alice', 'John', 'Eve', 'Charlie', 'Diana', 'Frank'],
            'age': [25, 30, 45, 27, 22, 25, None, 33, 41, 38]
        })
    
    def test_create_completeness_rule(self):
        """Test creating a completeness rule with the helper function."""
        rule = create_completeness_rule('test_rule', ['name'], threshold=0.9)
        self.assertIsInstance(rule, CompletenessRule)
        self.assertEqual(rule.rule_id, 'test_rule')
        self.assertEqual(rule.columns, ['name'])
        self.assertEqual(rule.threshold, 0.9)
    
    def test_create_uniqueness_rule(self):
        """Test creating a uniqueness rule with the helper function."""
        rule = create_uniqueness_rule('test_rule', ['id'], threshold=1.0)
        self.assertIsInstance(rule, UniquenessRule)
        self.assertEqual(rule.rule_id, 'test_rule')
        self.assertEqual(rule.columns, ['id'])
        self.assertEqual(rule.threshold, 1.0)
    
    def test_create_value_range_rule(self):
        """Test creating a value range rule with the helper function."""
        rule = create_value_range_rule('test_rule', 'age', min_value=18, max_value=65)
        self.assertIsInstance(rule, ValueRangeRule)
        self.assertEqual(rule.rule_id, 'test_rule')
        self.assertEqual(rule.column, 'age')
        self.assertEqual(rule.min_value, 18)
        self.assertEqual(rule.max_value, 65)
    
    def test_create_pattern_match_rule(self):
        """Test creating a pattern match rule with the helper function."""
        rule = create_pattern_match_rule('test_rule', 'email', 
                                          pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.assertIsInstance(rule, PatternMatchRule)
        self.assertEqual(rule.rule_id, 'test_rule')
        self.assertEqual(rule.column, 'email')
        self.assertEqual(rule.pattern, r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def test_create_statistical_rule(self):
        """Test creating a statistical rule with the helper function."""
        rule = create_statistical_rule('test_rule', 'age', 'mean', 30, 5)
        self.assertIsInstance(rule, StatisticalRule)
        self.assertEqual(rule.rule_id, 'test_rule')
        self.assertEqual(rule.column, 'age')
        self.assertEqual(rule.stat_type, 'mean')
        self.assertEqual(rule.expected_value, 30)
        self.assertEqual(rule.tolerance, 5)


if __name__ == '__main__':
    unittest.main()