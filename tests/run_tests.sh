#!/bin/bash

# Exit on error
set -e

# Load environment variables
if [ -f tests/.env.test ]; then
    export $(cat tests/.env.test | grep -v '^#' | xargs)
fi

# Create test directories if they don't exist
mkdir -p test_results/coverage
mkdir -p test_results/benchmarks
mkdir -p test_results/reports

# Run tests with coverage and generate reports
pytest \
    --verbose \
    --cov=quantum_autoencoder \
    --cov-report=term-missing \
    --cov-report=html:test_results/coverage \
    --html=test_results/reports/report.html \
    --self-contained-html \
    --benchmark-only \
    --benchmark-group-by=func \
    --benchmark-warmup=true \
    --benchmark-min-rounds=10 \
    --randomly-seed=42 \
    --timeout=300 \
    --reruns=3 \
    --reruns-delay=1 \
    tests/ \
    "$@"

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "✅ All tests passed successfully!"
    echo "📊 Test reports are available in:"
    echo "   - Coverage HTML: test_results/coverage/index.html"
    echo "   - Test Report: test_results/reports/report.html"
else
    echo "❌ Some tests failed. Check the reports for details:"
    echo "   - Coverage HTML: test_results/coverage/index.html"
    echo "   - Test Report: test_results/reports/report.html"
    exit 1
fi 