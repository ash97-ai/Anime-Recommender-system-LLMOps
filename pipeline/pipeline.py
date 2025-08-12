from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY,MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self,persist_dir='chroma_db'):
        try:
            logger.info("Initializing AnimeRecommendationPipeline")

            vector_builder = VectorStoreBuilder(csv_path="",persist_dir=persist_dir)
            retriever = vector_builder.load_vectorstore().as_retriever()
            self.recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)
            logger.info("AnimeRecommendationPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AnimeRecommendationPipeline: {e}")
            raise CustomException("Failed to initialize AnimeRecommendationPipeline",e) from e
    
    def recommend(self,query:str):
        try:
            logger.info(f"Generating recommendations for query: {query}")
            recommendations = self.recommender.get_recommendation(query)
            logger.info("Recommendations generated successfully")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise CustomException("Failed to generate recommendations",e) from e
