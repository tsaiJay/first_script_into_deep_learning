# tool code for sequential training

echo "script begin..."

python train_test.py --c _config_resnet7.yaml 
wait
python train_test.py --c _config_resnet4.yaml 
wait

echo "script end..."