echo
echo "Imbalance : 2:98"
echo
python3 smoteboost.py DecisionTree 0.02 4 10000 30 0.3
echo
echo "Imbalance : 4:96"
echo
python3 smoteboost.py DecisionTree 0.04 4 10000 30 0.3
echo
echo "Imbalance : 8:92"
echo
python3 smoteboost.py DecisionTree 0.08 4 10000 30 0.3
echo
echo "Imbalance : 16:84"
echo
python3 smoteboost.py DecisionTree 0.16 4 10000 30 0.3
echo
echo "Imbalance : 32:68"
echo
python3 smoteboost.py DecisionTree 0.32 4 10000 30 0.3