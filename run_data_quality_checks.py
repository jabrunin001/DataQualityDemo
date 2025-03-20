#!/usr/bin/env python3
"""
Example script to run automated data quality checks on a dataset.

This script demonstrates how to use the data quality framework to:
1. Load data from CSV files or SQL queries
2. Apply data quality rules from a configuration file
3. Generate validation reports and visualizations
4. Send notifications based on validation results

It can be run as a standalone script or integrated into data pipelines.
"""

import os
import argparse
import pandas as pd
import json
from datetime import datetime
import logging
import sys
from typing import Dict, Any, Optional

# Import the data quality framework
from data_quality import DataQualityFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dq_check.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dq-check")


def load_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data, or None if loading failed
    """
    try:
        logger.info(f"Loading data from CSV: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        return None


def load_data_from_sql(connection_string: str, query: str) -> Optional[pd.DataFrame]:
    """
    Load data from a SQL query.
    
    Args:
        connection_string: Database connection string
        query: SQL query to execute
        
    Returns:
        DataFrame containing the query results, or None if query failed
    """
    try:
        import sqlalchemy
        
        logger.info(f"Connecting to database and executing query")
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {str(e)}")
        return None


def generate_dashboard(report_data: Dict[str, Any], output_path: str) -> None:
    """
    Generate a visual dashboard from validation results.
    
    Args:
        report_data: Validation report data
        output_path: Path to save the dashboard HTML
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a DataFrame from the results for easier visualization
    results = []
    for result in report_data.get("results", []):
        results.append({
            "rule_id": result.get("rule_id", "unknown"),
            "success": result.get("success", False),  # Ensure success is included
            "severity": result.get("severity", "medium"),
            "message": result.get("message", "")
        })
    
    results_df = pd.DataFrame(results)
    
    # Check if 'success' column exists
    if 'success' not in results_df.columns:
        logger.error("No 'success' column found in results DataFrame.")
        return
    
    # Set up the dashboard
    plt.figure(figsize=(12, 10))
    
    # 1. Overall pass/fail pie chart
    plt.subplot(2, 2, 1)
    success_counts = results_df["success"].value_counts()
    plt.pie(
        success_counts,
        labels=["Pass", "Fail"],
        colors=["#28a745", "#dc3545"],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Overall Pass Rate")
    
    # 2. Results by severity
    plt.subplot(2, 2, 2)
    severity_counts = pd.crosstab(results_df["severity"], results_df["success"])
    severity_counts.plot(
        kind="barh",
        color=["#28a745", "#dc3545"],
        ax=plt.gca()
    )
    plt.title("Results by Severity")
    plt.xlabel("Count")
    plt.ylabel("Severity")
    plt.legend(["Pass", "Fail"])
    
    # 3. Rule results table
    plt.subplot(2, 1, 2)
    plt.axis('off')
    
    # Create a table of results
    table_data = []
    for _, row in results_df.iterrows():
        status = "✅" if row["success"] else "❌"
        # Truncate long messages
        message = row["message"]
        if len(message) > 50:
            message = message[:47] + "..."
        
        table_data.append([
            row["rule_id"],
            status,
            row["severity"],
            message
        ])
    
    table = plt.table(
        cellText=table_data,
        colLabels=["Rule ID", "Status", "Severity", "Message"],
        loc="center",
        cellLoc="left",
        colColours=["#f8f9fa"] * 4,
        colWidths=[0.15, 0.1, 0.15, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title("Rule Results", pad=30)
    
    # Add dashboard title and metadata
    summary = report_data.get("summary", {})
    validation_id = summary.get("validation_id", "unknown")
    timestamp = summary.get("timestamp", datetime.now().isoformat())
    if isinstance(timestamp, str):
        timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    plt.suptitle(
        f"Data Quality Validation Report: {validation_id}\n"
        f"Generated: {timestamp_str}",
        fontsize=16, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    logger.info(f"Generated dashboard at {output_path}")


def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run data quality checks on a dataset')
    parser.add_argument('--config', required=True, help='Path to data quality configuration file')
    parser.add_argument('--data-source', required=True, help='Path to CSV file or SQL connection string')
    parser.add_argument('--sql-query', help='SQL query to execute (if data source is a connection string)')
    parser.add_argument('--group', help='Rule group to validate against (if not specified, use all rules)')
    parser.add_argument('--output-dir', default='./validation_reports', help='Directory for validation reports')
    parser.add_argument('--dashboard', action='store_true', help='Generate a visual dashboard')
    parser.add_argument('--notify', action='store_true', help='Send notification with results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    try:
        framework = DataQualityFramework(args.config)
        logger.info(f"Loaded data quality configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return 1
    
    # Load data
    if args.sql_query:
        # Load from SQL
        data = load_data_from_sql(args.data_source, args.sql_query)
    else:
        # Load from CSV
        data = load_data_from_csv(args.data_source)
    
    if data is None:
        logger.error("Failed to load data, aborting")
        return 1
    
    # Generate validation ID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    validation_id = f"validation-{timestamp}"
    
    # Create report path
    report_path = os.path.join(args.output_dir, f"{validation_id}.json")
    
    # Run validation
    logger.info(f"Running validation {validation_id}")
    validation_report = framework.validate_data(
        data,
        group_name=args.group,
        validation_id=validation_id,
        save_report=True,
        report_path=report_path
    )
    
    # Print summary
    summary = validation_report["summary"]
    logger.info(f"Validation completed: {summary['overall_status']}")
    logger.info(f"  Total Rules: {summary['total_rules']}")
    logger.info(f"  Passed Rules: {summary['passed_rules']}")
    logger.info(f"  Failed Rules: {summary['failed_rules']}")
    
    # Generate dashboard if requested
    if args.dashboard:
        dashboard_path = os.path.join(args.output_dir, f"{validation_id}-dashboard.png")
        generate_dashboard(validation_report, dashboard_path)
    
    # Send notification if requested
    if args.notify:
        # Create a validator to access notification method
        from data_quality import DataQualityValidation
        validator = DataQualityValidation([])
        
        # Set validation results
        validator.validation_id = validation_id
        validator.start_time = datetime.now()
        validator.end_time = datetime.now()
        validator.results = validation_report["results"]
        
        # Get notification settings from config
        try:
            with open(args.config, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
            
            notification_config = config.get("settings", {}).get("notification", {})
            
            # Send notification
            if notification_config.get("email_enabled", False):
                logger.info("Sending notification email")
                validator.send_notification(
                    email_to=notification_config.get("email_to", []),
                    email_from=notification_config.get("email_from", ""),
                    smtp_server=notification_config.get("smtp_server", ""),
                    smtp_port=notification_config.get("smtp_port", 587),
                    username=notification_config.get("smtp_username"),
                    password=notification_config.get("smtp_password"),
                    only_on_failures=notification_config.get("notify_on_failures_only", True)
                )
            else:
                logger.info("Email notifications not enabled in config")
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")
    
    # Return status code
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())