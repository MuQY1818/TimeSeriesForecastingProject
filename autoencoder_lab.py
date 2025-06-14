from probe_forecaster_utils import train_probe_forecaster

def main():
    """
    Main function to run the training and visualization of the ProbeForecaster.
    This script serves as the primary entry point for experimenting with and
    training the forecaster model.
    """
    config = {
        'seq_len': 365,
        'd_model': 64,
        'nhead': 4,
        'num_agents': 8,
        'num_lags': 14,
        'epochs': 150,
        'patience': 25,
        'batch_size': 32,
        'learning_rate': 0.0005
    }
    
    model_path = "pretrained_models/probe_forecaster.pth"
    print("\n" + "="*80)
    print("🚀 训练 ProbeForecaster 并保存到", model_path)
    print("="*80)

    # 只负责训练并保存权重
    train_probe_forecaster(config, model_path)

    print("✅ ProbeForecaster 训练完成！")

if __name__ == "__main__":
    main() 