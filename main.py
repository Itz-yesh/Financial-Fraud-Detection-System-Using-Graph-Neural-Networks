"""
Main Execution Script
End-to-end pipeline for fraud detection using GNNs
"""

import os
import argparse
import torch

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.checkpoint import CheckpointManager
from src.data_preprocessing.data_loader import DataLoader
from src.data_preprocessing.preprocessor import DataPreprocessor
from src.graph_construction.graph_builder import GraphBuilder
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator
from src.explainability.attention_visualizer import AttentionVisualizer
from src.explainability.node_importance import NodeImportanceCalculator
from src.explainability.graph_visualizer import GraphVisualizer


def main(args):
    """Main execution function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger
    logger = get_logger(
        experiment_name=config.get('experiment.name', 'fraud_detection_gnn'),
        log_dir=config.get('experiment.log_dir', 'results/logs'),
        use_tensorboard=config.get('experiment.tensorboard_logging', True)
    )
    
    logger.info("="*70)
    logger.info("FINANCIAL FRAUD DETECTION USING GRAPH NEURAL NETWORKS")
    logger.info("="*70)
    
    # Apply CLI overrides directly into config (env vars are NOT read by config_loader)
    if args.dataset:
        config.update('data.dataset_name', args.dataset)
    if args.epochs:
        config.update('training.num_epochs', int(args.epochs))

    # 1. Load Data
    logger.info("\n[1/7] Loading dataset...")
    dataset_name = config.get('data.dataset_name', 'paysim')
    logger.info(f"Dataset selected: {dataset_name}")
    data_loader = DataLoader(
        dataset_name=dataset_name,
        data_dir=config.get('data.raw_data_path', 'data/raw')
    )
    df = data_loader.load_data()
    logger.info(f"Loaded {len(df)} transactions")
    
    # 2. Preprocess Data
    logger.info("\n[2/7] Preprocessing data...")
    preprocessor = DataPreprocessor(
        train_ratio=config.get('data.train_ratio', 0.7),
        val_ratio=config.get('data.val_ratio', 0.15),
        test_ratio=config.get('data.test_ratio', 0.15),
        random_seed=config.get('data.random_seed', 42)
    )
    
    df_processed, metadata = preprocessor.fit_transform(df)
    train_df, val_df, test_df = preprocessor.split_data(df_processed)
    
    # Add split column
    df_processed['split'] = 'train'
    df_processed.loc[val_df.index, 'split'] = 'val'
    df_processed.loc[test_df.index, 'split'] = 'test'
    
    # 3. Build Graph
    logger.info("\n[3/7] Constructing transaction graph...")
    graph_builder = GraphBuilder(
        temporal_edges=config.get('graph.temporal_edges', True),
        time_window_hours=config.get('graph.time_window_hours', 24)
    )
    
    feature_columns = metadata['feature_columns']
    graph_data = graph_builder.build_graph(df_processed, feature_columns)
    
    logger.info(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # 4. Create Model
    logger.info("\n[4/7] Creating GNN model...")
    model_name = config.get('model.architecture', 'jump_attention')
    model = create_model(
        model_name=model_name,
        in_channels=graph_data.num_node_features,
        config=config['model']
    )
    
    logger.log_model_architecture(model)
    
    # 5. Train Model
    logger.info("\n[5/7] Training model...")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.get('experiment.checkpoint_dir', 'experiments/checkpoints')
    )
    
    if args.mode == 'train':
        trainer = Trainer(
            model=model,
            data=graph_data,
            config=config['training'],
            logger=logger,
            checkpoint_manager=checkpoint_manager
        )
        
        history = trainer.train(num_epochs=config.get('training.num_epochs', 100))
    else:
        logger.info(f"Skipping training (mode={args.mode}). Using existing checkpoint.")
    
    # 6. Evaluate Model
    logger.info("\n[6/7] Evaluating model...")
    
    # Load best model
    checkpoint_manager.load_checkpoint(model, checkpoint_path=None)
    
    # Get device from model parameters
    device = str(next(model.parameters()).device)
    evaluator = ModelEvaluator(
        model=model,
        data=graph_data,
        device=device
    )
    
    results = evaluator.generate_report(output_dir='results')
    
    # Print test results
    test_metrics = results['test']['metrics']
    logger.info("\nTest Set Performance:")
    logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    
    # 7. Generate Explanations
    if config.get('explainability.enable_attention_viz', True):
        logger.info("\n[7/7] Generating explanations...")
        
        # Visualize attention
        if model_name in ['gat', 'jump_attention']:
            attention_viz = AttentionVisualizer(model, graph_data)
            
            # Find some fraud cases to explain
            fraud_indices = torch.where(graph_data.y == 1)[0][:5].tolist()
            attention_viz.explain_predictions(
                fraud_indices,
                output_dir='results/explanations'
            )
        
        # Node importance
        importance_calc = NodeImportanceCalculator(model, graph_data)
        fraud_idx = torch.where(graph_data.y == 1)[0][0].item()
        explanation = importance_calc.explain_prediction(fraud_idx)
        
        logger.info(f"\nExample Explanation (Node {fraud_idx}):")
        logger.info(f"  Predicted: {explanation['predicted_label']} ({explanation['predicted_probability']:.3f})")
        logger.info(f"  True Label: {explanation['true_label']}")
        logger.info(f"  Top Features: {explanation['top_features'][:3]}")
        
        # Graph visualization
        graph_viz = GraphVisualizer(graph_data)
        graph_viz.visualize_fraud_clusters(save_path='results/fraud_clusters.png')
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: results/")
    logger.info(f"Model checkpoints saved to: {config.get('experiment.checkpoint_dir')}")
    
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Financial Fraud Detection using GNNs')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'test'],
                       help='Execution mode')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load config first so we can apply CLI overrides directly into it
    # NOTE: env vars were previously set here but config_loader.py never reads them,
    # so overrides had no effect. Now we pass args directly and apply them in main().
    main(args)
