# Automated Data Quality Checks Framework

A comprehensive, extensible framework for defining, executing, and reporting on data quality checks across datasets.

## Business Value

Data quality issues create significant business risks:

1. **Decision Risk**: Poor data quality leads to incorrect analysis and flawed decisions
2. **Pipeline Reliability**: Undetected quality issues silently corrupt downstream systems
3. **Trust Erosion**: Stakeholders lose confidence when quality issues appear
4. **Remediation Cost**: Fixing quality issues becomes exponentially more expensive when detected late
5. **Compliance Exposure**: Many regulations mandate data accuracy and auditability

This framework creates business value by:

- **Early Detection**: Identify issues at their source before they propagate
- **Consistent Evaluation**: Apply unified quality standards across systems
- **Automated Assessment**: Reduce manual review effort through automation
- **Traceable Validation**: Create audit trails of quality checks
- **Proactive Alerting**: Enable immediate notification when issues emerge

## Core Architecture

The framework is built around these key components:

1. **DataQualityRule**: Base class for all quality rules with common evaluation interface
2. **Rule Implementations**: Specialized rules for different quality dimensions (completeness, uniqueness, etc.)
3. **DataQualityValidation**: Executes rules and collects results
4. **DataQualityFramework**: Manages rule configuration and execution
5. **Reporting & Visualization**: Generates reports and dashboards from results

## Quality Dimensions

The framework evaluates data across multiple quality dimensions:

| Dimension | Definition | Example Rules |
|-----------|------------|---------------|
| **Completeness** | Presence of required data | Check for null values in critical columns |
| **Uniqueness** | Absence of duplicates | Ensure primary keys are unique |
| **Consistency** | Logical coherence across fields | Check that end dates are after start dates |
| **Validity** | Conformance to defined rules | Validate that values match expected patterns |
| **Accuracy** | Correctness of values | Verify that calculated totals match source data |
| **Integrity** | Correctness of relationships | Ensure foreign keys exist in referenced tables |
| **Timeliness** | Recency of data | Check that data is updated within required timeframes |

## Supported Rule Types

The framework includes these rule implementations:

1. **CompletenessRule**: Check for presence of values (absence of nulls)
2. **UniquenessRule**: Check for duplicate values in columns
3. **ValueRangeRule**: Verify values fall within expected numeric ranges
4. **PatternMatchRule**: Validate text values against regex patterns
5. **ReferentialIntegrityRule**: Ensure values exist in reference datasets
6. **StatisticalRule**: Check if statistical properties match expectations
7. **CustomSQLRule**: Execute custom SQL queries for validation
8. **CustomPythonRule**: Use custom Python functions for validation

## Installation

```bash
# Clone the repository
git clone 
cd data-quality-framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Define Rules in Code

```python
from data_quality import DataQualityFramework, create_completeness_rule, create_value_range_rule

# Create framework
framework = DataQualityFramework()

# Add rules
framework.add_rule(create_completeness_rule(
    'completeness_id', ['id'], 
    description="ID should never be null"
))

framework.add_rule(create_value_range_rule(
    'range_amount', 'amount', min_value=0, max_value=10000,
    description="Amount should be between 0 and 10,000"
))

# Load data
import pandas as pd
data = pd.read_csv('transactions.csv')

# Run validation
validation_report = framework.validate_data(data)

# Print summary
print(f"Status: {validation_report['summary']['overall_status']}")
print(f"Passed: {validation_report['summary']['passed_rules']}")
print(f"Failed: {validation_report['summary']['failed_rules']}")
```

### 2. Define Rules in Configuration

```yaml
# data_quality_config.yaml
rules:
  - id: "completeness_id"
    type: "completeness"
    description: "ID should never be null"
    columns: ["id"]
    threshold: 1.0
    severity: "critical"

  - id: "range_amount"
    type: "value_range"
    description: "Amount should be between 0 and 10,000"
    column: "amount"
    min_value: 0
    max_value: 10000
    severity: "high"
```

```python
from data_quality import DataQualityFramework

# Create framework from config
framework = DataQualityFramework('data_quality_config.yaml')

# Load data
import pandas as pd
data = pd.read_csv('transactions.csv')

# Run validation
validation_report = framework.validate_data(data)
```

### 3. Command-line Execution

```bash
# Run validation using the command-line tool
python run_data_quality_checks.py \
  --config data_quality_config.yaml \
  --data-source transactions.csv \
  --output-dir reports \
  --dashboard
```

## Example Validation Report

```json
{
  "summary": {
    "validation_id": "validation-20230615-123456",
    "timestamp": "2023-06-15T12:34:56.789012",
    "duration_seconds": 1.25,
    "total_rules": 5,
    "passed_rules": 4,
    "failed_rules": 1,
    "pass_rate": 0.8,
    "critical_failures": 0,
    "high_failures": 1,
    "overall_status": "FAIL"
  },
  "results": [
    {
      "rule_id": "completeness_id",
      "description": "ID should never be null",
      "severity": "critical",
      "success": true,
      "message": "All columns meet completeness threshold",
      "metrics": {
        "columns_checked": 1,
        "columns_failed": 0,
        "completion_rates": {"id": 1.0},
        "average_completion_rate": 1.0
      },
      "details": {
        "failing_columns": [],
        "threshold": 1.0
      }
    },
    {
      "rule_id": "range_amount",
      "description": "Amount should be between 0 and 10,000",
      "severity": "high",
      "success": false,
      "message": "3 values in column 'amount' are outside the specified range (99.40% in range)",
      "metrics": {
        "in_range_rate": 0.9940,
        "out_of_range_count": 3
      },
      "details": {
        "total_values": 500,
        "in_range_count": 497,
        "min_value": 0,
        "max_value": 10000,
        "include_min": true,
        "include_max": true,
        "out_of_range_stats": {
          "min": 10200.0,
          "max": 15000.0,
          "mean": 12100.0,
          "median": 11100.0,
          "examples": [10200.0, 11100.0, 15000.0]
        }
      }
    }
  ]
}
```

## Key Features

### Rule Groups

Group related rules for specific validation contexts:

```python
# Create rule groups
framework.create_rule_group('transaction_validation', [
    'completeness_transaction_id',
    'range_amount',
    'pattern_transaction_type'
])

# Validate using a specific group
validation_report = framework.validate_data(data, group_name='transaction_validation')
```

### Visualization

Generate visual dashboards from validation results:

```python
# Create validation object
validator = DataQualityValidation(rules)

# Execute validation
validator.execute(data)

# Generate dashboard
validator.generate_dashboard('validation_dashboard.png')
```

### Notifications

Send email notifications with validation results:

```python
# Send notification
validator.send_notification(
    email_to=['data.team@example.com'],
    email_from='data.quality@example.com',
    smtp_server='smtp.example.com',
    only_on_failures=True
)
```

## Extending the Framework

### Creating Custom Rules

Implement specialized validation logic by creating custom rules:

```python
from data_quality import DataQualityRule

class MyCustomRule(DataQualityRule):
    def __init__(self, rule_id, description, severity='medium'):
        super().__init__(rule_id, description, severity)
        
    def evaluate(self, data, **kwargs):
        # Implement custom validation logic here
        success = True
        message = "Validation passed"
        
        return {
            "success": success,
            "message": message,
            "metrics": {
                "custom_metric": 100
            },
            "details": {
                "additional_info": "Custom validation details"
            }
        }
```

### Custom Python Functions

Use the CustomPythonRule for one-off validations without creating new rule classes:

```python
def check_date_ranges(data, **kwargs):
    # Check that end_date is after start_date
    invalid_ranges = (data['end_date'] <= data['start_date']).sum()
    success = invalid_ranges == 0
    
    return {
        "success": success,
        "message": f"Found {invalid_ranges} invalid date ranges",
        "metrics": {"invalid_count": invalid_ranges}
    }

# Create rule using the function
framework.add_rule(create_custom_python_rule(
    'date_range_check', check_date_ranges,
    description="Check that end dates are after start dates"
))
```

## Integration Patterns

### Data Pipeline Integration

Integrate quality checks into data processing pipelines:

1. **Pre-Processing Validation**: Validate raw data before transformation
2. **Post-Processing Validation**: Verify data integrity after transformation
3. **Decision Logic**: Use validation results to route data flows
4. **Alert Triggers**: Send alerts when critical issues are detected

### ETL/ELT Integration

Integrate with ETL tools like Apache Airflow:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from data_quality import DataQualityFramework

def run_data_quality_checks(**kwargs):
    # Load framework
    framework = DataQualityFramework('dq_config.yaml')
    
    # Get data from previous task
    data = kwargs['ti'].xcom_pull(task_ids='extract_data')
    
    # Run validation
    validation_report = framework.validate_data(data)
    
    # Fail the task if validation failed
    if validation_report['summary']['overall_status'] == 'FAIL':
        raise ValueError("Data quality validation failed")
    
    return validation_report

with DAG('etl_with_quality_checks',
         start_date=datetime(2023, 1, 1),
         schedule_interval='@daily') as dag:
    
    # Extract task
    extract_task = PythonOperator(...)
    
    # Data quality check task
    quality_check_task = PythonOperator(
        task_id='check_data_quality',
        python_callable=run_data_quality_checks
    )
    
    # Transform task
    transform_task = PythonOperator(...)
    
    # Define task dependencies
    extract_task >> quality_check_task >> transform_task
```

## Best Practices

1. **Start Simple**: Begin with critical quality checks and expand gradually
2. **Use Appropriate Severity**: Assign severity levels based on business impact
3. **Group Related Rules**: Create rule groups for specific validation contexts
4. **Version Control Configurations**: Keep rule configurations in version control
5. **Monitor False Positives**: Refine rules that generate excessive alerts
6. **Progressive Validation**: Apply different rule sets at different pipeline stages
7. **Document Rules**: Add clear descriptions explaining each rule's purpose
