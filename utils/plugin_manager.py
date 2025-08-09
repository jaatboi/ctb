import os
import importlib.util
import inspect
from typing import Dict, List, Type, Any, Callable
from interfaces.base_components import BaseDataCollector, BaseTechnicalAnalyzer, BaseSignalGenerator, BaseSentimentAnalyzer

class PluginManager:
    """Manages loading and registration of plugins"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self.indicators: Dict[str, Callable] = {}
        self.strategies: Dict[str, Type[BaseSignalGenerator]] = {}
        self.data_sources: Dict[str, Type[BaseDataCollector]] = {}
        self.sentiment_sources: Dict[str, Type[BaseSentimentAnalyzer]] = {}
        
        # Create plugin directories if they don't exist
        for directory in self.plugin_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Load plugins
        self.load_plugins()
    
    def load_plugins(self):
        """Load all plugins from plugin directories"""
        for directory in self.plugin_dirs:
            self._load_plugins_from_directory(directory)
    
    def _load_plugins_from_directory(self, directory: str):
        """Load plugins from a specific directory"""
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                file_path = os.path.join(directory, filename)
                
                # Load module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Register plugins
                self._register_plugins(module)
    
    def _register_plugins(self, module):
        """Register plugins from a loaded module"""
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, '_is_indicator'):
                # Register indicator function
                indicator_name = getattr(obj, '_indicator_name', name)
                self.indicators[indicator_name] = obj
                print(f"Registered indicator: {indicator_name}")
            
            elif inspect.isclass(obj):
                if issubclass(obj, BaseSignalGenerator) and obj != BaseSignalGenerator:
                    # Register strategy class
                    strategy_name = getattr(obj, '_strategy_name', name)
                    self.strategies[strategy_name] = obj
                    print(f"Registered strategy: {strategy_name}")
                
                elif hasattr(obj, '_is_data_source') and obj != BaseDataCollector:
                    # Register data source class
                    source_name = getattr(obj, '_source_name', name)
                    self.data_sources[source_name] = obj
                    print(f"Registered data source: {source_name}")
                
                elif hasattr(obj, '_is_sentiment_source') and obj != BaseSentimentAnalyzer:
                    # Register sentiment source class
                    source_name = getattr(obj, '_source_name', name)
                    self.sentiment_sources[source_name] = obj
                    print(f"Registered sentiment source: {source_name}")
    
    def get_indicator(self, name: str) -> Callable:
        """Get an indicator function by name"""
        if name in self.indicators:
            return self.indicators[name]
        raise ValueError(f"Indicator '{name}' not found")
    
    def get_strategy(self, name: str) -> BaseSignalGenerator:
        """Get a strategy class by name"""
        if name in self.strategies:
            return self.strategies[name]()
        raise ValueError(f"Strategy '{name}' not found")
    
    def get_data_source(self, name: str) -> BaseDataCollector:
        """Get a data source class by name"""
        if name in self.data_sources:
            return self.data_sources[name]()
        raise ValueError(f"Data source '{name}' not found")
    
    def get_sentiment_source(self, name: str) -> BaseSentimentAnalyzer:
        """Get a sentiment source class by name"""
        if name in self.sentiment_sources:
            return self.sentiment_sources[name]()
        raise ValueError(f"Sentiment source '{name}' not found")
    
    def list_indicators(self) -> List[str]:
        """List all available indicators"""
        return list(self.indicators.keys())
    
    def list_strategies(self) -> List[str]:
        """List all available strategies"""
        return list(self.strategies.keys())
    
    def list_data_sources(self) -> List[str]:
        """List all available data sources"""
        return list(self.data_sources.keys())
    
    def list_sentiment_sources(self) -> List[str]:
        """List all available sentiment sources"""
        return list(self.sentiment_sources.keys())


# Decorators for plugin registration
def indicator(name: str = None):
    """Decorator to register a function as an indicator"""
    def decorator(func):
        func._is_indicator = True
        func._indicator_name = name or func.__name__
        return func
    return decorator


def strategy(name: str = None):
    """Decorator to register a class as a strategy"""
    def decorator(cls):
        cls._is_strategy = True
        cls._strategy_name = name or cls.__name__
        return cls
    return decorator


def data_source(name: str = None):
    """Decorator to register a class as a data source"""
    def decorator(cls):
        cls._is_data_source = True
        cls._source_name = name or cls.__name__
        return cls
    return decorator


def sentiment_source(name: str = None):
    """Decorator to register a class as a sentiment source"""
    def decorator(cls):
        cls._is_sentiment_source = True
        cls._source_name = name or cls.__name__
        return cls
    return decorator
