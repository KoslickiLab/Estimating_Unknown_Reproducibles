python ./build_ext.py build
mkdir exec_ext_files
mv ./build/lib.*/pyx_files/* exec_ext_files/
rm -r ./build