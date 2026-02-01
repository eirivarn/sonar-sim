"""Simple profiling utilities to identify performance bottlenecks."""

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    """Lightweight profiler for tracking execution time."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.enabled = True
    
    @contextmanager
    def measure(self, name):
        """Context manager to measure execution time.
        
        Usage:
            with profiler.measure('sonar_scan'):
                sonar.scan(grid)
        """
        if not self.enabled:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)
            self.call_counts[name] += 1
    
    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
        self.call_counts.clear()
    
    def report(self, min_time=0.001, top_n=10):
        """Print profiling report.
        
        Args:
            min_time: Only show operations taking more than this (seconds)
            top_n: Show top N slowest operations
        """
        if not self.timings:
            print("No profiling data collected.")
            return
        
        # Calculate statistics
        stats = []
        for name, times in self.timings.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            max_time = max(times) if times else 0
            min_time_val = min(times) if times else 0
            
            if total >= min_time:
                stats.append({
                    'name': name,
                    'total': total,
                    'count': count,
                    'avg': avg,
                    'max': max_time,
                    'min': min_time_val
                })
        
        # Sort by total time
        stats.sort(key=lambda x: x['total'], reverse=True)
        
        # Print report
        print("\n" + "="*80)
        print("PERFORMANCE PROFILE")
        print("="*80)
        
        total_time = sum(s['total'] for s in stats)
        print(f"Total measured time: {total_time:.3f}s")
        print(f"Number of operations: {len(stats)}")
        print("="*80)
        
        # Column headers
        print(f"{'Operation':<30} {'Total(s)':>10} {'Calls':>8} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'%':>6}")
        print("-"*80)
        
        # Show top N
        for stat in stats[:top_n]:
            pct = (stat['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{stat['name']:<30} "
                  f"{stat['total']:>10.3f} "
                  f"{stat['count']:>8} "
                  f"{stat['avg']*1000:>10.2f} "
                  f"{stat['min']*1000:>10.2f} "
                  f"{stat['max']*1000:>10.2f} "
                  f"{pct:>5.1f}%")
        
        print("="*80 + "\n")
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable profiling (no overhead)."""
        self.enabled = False


# Global profiler instance
_global_profiler = Profiler()


def get_profiler():
    """Get the global profiler instance."""
    return _global_profiler


def measure(name):
    """Decorator to measure function execution time.
    
    Usage:
        @measure('my_function')
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _global_profiler.measure(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
