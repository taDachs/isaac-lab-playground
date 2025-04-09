# folders that contain extensions (list in bash)
ISAACSIM_ROOT_PATH=/isaac-sim
ISAACLAB_PATH=/home/ubuntu/IsaacLab
unset PYTHONPATH
for folder in exts extsDeprecated extsPhysics extscache; do
  for ext in $(ls ${ISAACSIM_ROOT_PATH}/${folder}); do
    export PYTHONPATH=${ISAACSIM_ROOT_PATH}/${folder}/${ext}:${PYTHONPATH}
  done
done

# add python_packages as well
export PYTHONPATH=${ISAACLAB_PATH}/python_packages:${PYTHONPATH}

for ext in $(ls ${ISAACLAB_PATH}/source); do
  export PYTHONPATH=${ISAACLAB_PATH}/source/${ext}:${PYTHONPATH}
done

export PYTHONPATH=${ISAACSIM_ROOT_PATH}/kit/python/lib/python3.10/site-packages/:${PYTHONPATH}:
export PYTHONPATH=$(pwd)/source/isaac_lab_playground:${PYTHONPATH}
export PYTHONPATH=$(pwd)/skrl:${PYTHONPATH}
