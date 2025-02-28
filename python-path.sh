# folders that contain extensions (list in bash)
EXTENSION_FOLDERS="exts extsDeprecated extsPhysics extscache"


for folder in $EXTENSION_FOLDERS; do
  for ext in $(ls ${ISAACSIM_ROOT_PATH}/${folder}); do
    export PYTHONPATH=${ISAACSIM_ROOT_PATH}/${folder}/${ext}:${PYTHONPATH}
  done
done

# add python_packages as well
export PYTHONPATH=${ISAACLAB_PATH}/python_packages:${PYTHONPATH}

for ext in $(ls ${ISAACLAB_PATH}/source); do
  export PYTHONPATH=${ISAACLAB_PATH}/source/${ext}:${PYTHONPATH}
done
