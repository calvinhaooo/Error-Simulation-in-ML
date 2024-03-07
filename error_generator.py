import jenga
from jenga.tasks.income import IncomeEstimationTask

task = IncomeEstimationTask(seed=42, ignore_incomplete_records_for_training=True)
