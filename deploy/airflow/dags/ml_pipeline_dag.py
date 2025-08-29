from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.data_download import download_and_unzip
from src.data_extraction import data_extraction
from src.data_preprocessing import preprocess_data
from src.training import model_training
from src.evaluation import model_evaluation
def create_dag():
    with DAG(
        dag_id="ml_pipeline",
        start_date=datetime(2022, 1, 1),
        schedule=None,
        catchup=False,
        tags=["ml"],
    ) as dag:
        print('Creating ML Pipeline DAG')
        download_data_task = PythonOperator(
            task_id="download_data",
            python_callable=download_and_unzip,
        )
        clean_extraction_task = PythonOperator(
            task_id="clean_extraction",
            python_callable=data_extraction,
            op_kwargs={
            'argv': [
                '--use-default'
            ]
            }
        )
        preprocess_data_task = PythonOperator(
          task_id="data_preprocessing",
          python_callable=preprocess_data,
          op_kwargs={
              'argv': [
                  '--input', '/app/data/raw/MWI_2010_individual.dta',
                  '--input-format', 'dta',
                  '--output', '/app/data/preprocessed/MWI_2010_individual_processed.csv'
              ]
          },
        )
        model_training_task = PythonOperator(
            task_id="model_training",
            python_callable=model_training,
            op_kwargs={
                'argv': [
                    '--input', '/app/data/preprocessed/MWI_2010_individual_processed.csv',
                    '--input-format', 'csv',
                    '--output', '/app/models'
                ]
            },
        )
        model_evaluation_task = PythonOperator(
            task_id="model_evaluation",
            python_callable=model_evaluation,
            op_kwargs={
                'argv': [
                    '--input', '/app/data/preprocessed/MWI_2010_individual_processed.csv',
                    '--input-format', 'csv',
                    '--models-dict', '/app/models',
                    '--output', '/app/reports'
                ]
            },
        )
        download_data_task >> clean_extraction_task >> \
        preprocess_data_task >> model_training_task >> \
            model_evaluation_task
        return dag


dag = create_dag()
