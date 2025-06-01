import re
import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNNModel(nn.Module):
    """改進版深度神經網路模型，兼容舊版本"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout_rate: float = 0.2, legacy_mode: bool = False):
        super(DNNModel, self).__init__()
        self.input_dim = input_dim
        self.legacy_mode = legacy_mode
        
        if legacy_mode:
            # 舊版本結構，兼容原有模型
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)
        else:
            # 新版本結構
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.legacy_mode:
            # 舊版本前向傳播
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
        else:
            # 新版本前向傳播
            return self.network(x)

class TextPreprocessor:
    """文本預處理類"""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """清理和預處理文本"""
        if not isinstance(text, str):
            text = str(text)
        
        # 保留更多有用信息，不只是移除標點
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # 移除URL
        text = re.sub(r'@\w+|#\w+', '', text)  # 移除@和#標籤
        text = re.sub(r'\d+', 'NUM', text)  # 數字替換為NUM
        text = re.sub(r'[^\w\s]', ' ', text)  # 移除標點但保留空格
        text = re.sub(r'\s+', ' ', text).strip()  # 清理多餘空格
        
        return text

class ChangePercentagePredictor:
    """改進版預測器類"""
    
    def __init__(self, model_path: Optional[str] = None):
        # 如果沒有指定路徑，使用當前工作目錄
        if model_path is None:
            self.model_path = Path.cwd()
        else:
            self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.tfidf = None
        self.sid = None
        self.preprocessor = TextPreprocessor()
        
    def load_models(
        self, 
        model_file: str = "dnn_model_change_percentage.pth",
        scaler_file: str = "scaler.pth",
        tfidf_file: str = "tfidf.pth", 
        sid_file: str = "sid.pth"
    ) -> bool:
        """加載所有必要的模型和預處理器"""
        try:
            # 載入DNN模型 - 檢查相對路徑和絕對路徑
            model_full_path = self.model_path / model_file
            
            # 如果指定路徑不存在，嘗試當前目錄
            if not model_full_path.exists():
                model_full_path = Path(model_file)
                logger.info(f"嘗試使用相對路徑: {model_full_path}")
            
            # 如果還是不存在，嘗試常見的模型目錄
            if not model_full_path.exists():
                possible_paths = [
                    Path.cwd() / model_file,
                    Path.cwd() / "models" / model_file,
                    Path.cwd().parent / model_file
                ]
                
                for path in possible_paths:
                    if path.exists():
                        model_full_path = path
                        logger.info(f"找到模型文件: {model_full_path}")
                        break
                else:
                    logger.error(f"在以下路徑都找不到模型文件 {model_file}:")
                    for path in [self.model_path / model_file] + possible_paths:
                        logger.error(f"  - {path}")
                    return False
            
            # 載入模型
            checkpoint = torch.load(model_full_path, map_location='cpu', weights_only=False)
            
            # 檢查是否為舊版本模型
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            is_legacy = 'fc1.weight' in state_dict  # 檢查是否有舊版本的層名稱
            
            if 'input_dim' in checkpoint:
                input_dim = checkpoint['input_dim']
            else:
                input_dim = 1003  # 預設值
                
            # 根據模型類型創建對應的模型
            self.model = DNNModel(input_dim, legacy_mode=is_legacy)
            
            if is_legacy:
                logger.info("檢測到舊版本模型結構，使用兼容模式載入")
            else:
                logger.info("使用新版本模型結構載入")
                
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"DNN模型載入成功: {model_full_path}")
            
            # 載入其他預處理器 - 使用相同的路徑查找邏輯
            def find_file(filename: str) -> Optional[Path]:
                """查找文件的通用函數"""
                possible_paths = [
                    self.model_path / filename,
                    Path(filename),
                    Path.cwd() / filename,
                    Path.cwd() / "models" / filename,
                    Path.cwd().parent / filename
                ]
                
                for path in possible_paths:
                    if path.exists():
                        return path
                return None
            
            # 載入 Scaler
            scaler_path = find_file(scaler_file)
            if scaler_path:
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Scaler載入成功: {scaler_path}")
                except:
                    # 如果joblib載入失敗，嘗試torch.load (兼容舊版本)
                    self.scaler = torch.load(scaler_path, map_location='cpu')
                    logger.info(f"Scaler載入成功 (使用torch.load): {scaler_path}")
            else:
                logger.error(f"找不到 Scaler 文件: {scaler_file}")
                return False
                
            # 載入 TF-IDF
            tfidf_path = find_file(tfidf_file)
            if tfidf_path:
                try:
                    self.tfidf = joblib.load(tfidf_path)
                    logger.info(f"TF-IDF載入成功: {tfidf_path}")
                except:
                    # 如果joblib載入失敗，嘗試torch.load (兼容舊版本)
                    self.tfidf = torch.load(tfidf_path, map_location='cpu')
                    logger.info(f"TF-IDF載入成功 (使用torch.load): {tfidf_path}")
            else:
                logger.error(f"找不到 TF-IDF 文件: {tfidf_file}")
                return False
                
            # 載入情感分析器
            sid_path = find_file(sid_file)
            if sid_path:
                try:
                    self.sid = joblib.load(sid_path)
                    logger.info(f"情感分析器載入成功: {sid_path}")
                except:
                    try:
                        self.sid = torch.load(sid_path, map_location='cpu', weights_only=False)
                        logger.info(f"情感分析器載入成功 (使用torch.load): {sid_path}")
                    except:
                        logger.warning("載入保存的情感分析器失敗，創建新的")
                        self.sid = SentimentIntensityAnalyzer()
            else:
                logger.info("沒有找到保存的情感分析器，創建新的")
                self.sid = SentimentIntensityAnalyzer()
            
            return True
            
        except Exception as e:
            logger.error(f"載入模型時發生錯誤: {str(e)}")
            return False
    
    def extract_features(self, text: str) -> np.ndarray:
        """提取文本特徵，保持與原始模型一致"""
        # 預處理文本
        preprocessed_text = self.preprocessor.preprocess_text(text)
        
        if not preprocessed_text.strip():
            logger.warning("預處理後的文本為空")
            preprocessed_text = "empty"
        
        # 情感分析
        sentiment_scores = self.sid.polarity_scores(preprocessed_text)
        sentiment = sentiment_scores['compound']
        
        # 原始的文本統計特徵（與舊版本保持一致）
        words = preprocessed_text.split()
        word_count = len(words)
        unique_word_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
        
        # TF-IDF特徵
        try:
            tfidf_features = self.tfidf.transform([preprocessed_text]).toarray()
            logger.info(f"TF-IDF特徵維度: {tfidf_features.shape[1]}")
        except Exception as e:
            logger.error(f"TF-IDF轉換錯誤: {str(e)}")
            # 如果出錯，使用預期的維度
            expected_tfidf_dim = 1000  # 預設值，會根據實際情況調整
            tfidf_features = np.zeros((1, expected_tfidf_dim))
        
        # 只使用原始的3個統計特徵（與原始模型保持一致）
        additional_features = np.array([[
            sentiment, 
            word_count, 
            unique_word_ratio
        ]])
        
        # 組合特徵
        features = np.hstack((tfidf_features, additional_features))
        
        logger.info(f"總特徵維度: {features.shape[1]} (TF-IDF: {tfidf_features.shape[1]}, 其他: {additional_features.shape[1]})")
        
        return features
    
    def predict(self, text: str) -> Dict[str, Any]:
        """進行預測並返回詳細結果"""
        if not all([self.model, self.scaler, self.tfidf, self.sid]):
            raise RuntimeError("模型尚未載入，請先調用load_models()")
        
        try:
            # 特徵提取
            features = self.extract_features(text)
            
            # 檢查特徵維度是否匹配
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 1003
            actual_features = features.shape[1]
            
            logger.info(f"期望特徵維度: {expected_features}, 實際特徵維度: {actual_features}")
            
            # 如果維度不匹配，進行調整
            if actual_features != expected_features:
                logger.warning(f"特徵維度不匹配，正在調整...")
                
                if actual_features > expected_features:
                    # 如果特徵太多，截斷
                    features = features[:, :expected_features]
                    logger.info(f"特徵已截斷至 {expected_features} 維")
                else:
                    # 如果特徵太少，填充零
                    padding = np.zeros((1, expected_features - actual_features))
                    features = np.hstack((features, padding))
                    logger.info(f"特徵已填充至 {expected_features} 維")
            
            # 特徵標準化
            features_scaled = self.scaler.transform(features)
            
            # 轉換為張量
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            
            # 預測
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction_value = prediction.item()
            
            # 計算置信度（簡單版本）
            confidence = min(abs(prediction_value) / 10.0, 1.0)  # 簡化的置信度計算
            
            return {
                'prediction': prediction_value,
                'confidence': confidence,
                'preprocessed_text': self.preprocessor.preprocess_text(text),
                'feature_count': features.shape[1]
            }
            
        except Exception as e:
            logger.error(f"預測時發生錯誤: {str(e)}")
            return {'error': str(e)}
    
    def batch_predict(self, texts: list) -> list:
        """批量預測"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main():
    """主函數示例"""
    # 創建預測器
    predictor = ChangePercentagePredictor()
    
    # 載入模型
    if not predictor.load_models():
        print("模型載入失敗，請檢查文件路徑")
        return
    
    print("=== 狗狗幣變化百分比預測器 ===")
    print("輸入 'quit' 結束程序")
    print("-" * 40)
    
    while True:
        try:
            text = input("\n請輸入文本以預測 Change_Percentage: ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("程序結束")
                break
            
            if not text.strip():
                print("請輸入有效的文本")
                continue
            
            # 進行預測
            result = predictor.predict(text)
            
            if 'error' in result:
                print(f"預測錯誤: {result['error']}")
            else:
                print(f"\n預測結果:")
                print(f"  Change_Percentage: {result['prediction']:.4f}")
                print(f"  置信度: {result['confidence']:.2f}")
                print(f"  特徵維度: {result['feature_count']}")
                
        except KeyboardInterrupt:
            print("\n程序被中斷")
            break
        except Exception as e:
            print(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()