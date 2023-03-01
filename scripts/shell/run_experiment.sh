initialPath=${initialPath:--1} ## /colmap_dataset/gerrard-hall-small-backup/
workspacePath=${workspacePath:--1} ## /colmap_dataset/gerrard-hall-small/
inputFormat=${inputFormat:-".txt"}
outputPath=${outputPath:--1}

delete=${delete:-0}
validate=${validate:-1}
align=${align:-1}
maxTryOutlier=${maxTryOutlier:-100}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

if [ "${initialPath}" == "-1" ];then
    printf "\Initial folder requiered (initialPath):\n"
    exit 1
fi
if [ "${workspacePath}" == "-1" ];then
    printf "\nWorkspace folder requiered (workspacePath):\n"
    exit 1
fi
if [ "${outputPath}" == "-1" ];then
    printf "\nOutput folder requiered (outputPath):\n"
    exit 1
fi

inputModel="${initialPath}/sparse/"
imagePath="${initialPath}/images/"
readDatabase="${initialPath}/database.db"
databasePath="${workspacePath}/database.db"
expOutputPath="${workspacePath}/sparse/"

stdNoiseMin=0.0
stdNoiseMax=3.0
stdNoiseStep=0.2

stdNoiseCounterMin=0 #Used for the while loop.
stdNoiseCounterMax=15 #Used for the while loop. = (stdNoiseMax - stdNoiseMin) / stdNoiseStep

stdNoise=${stdNoiseMin}
stdNoiseCounter=${stdNoiseCounterMin} #Used for the while loop.

outlierRatioMin=0.0
outlierRatioMax=0.5
outlierRatioStep=0.1

outlierRatioCounterMin=0 #Used for the while loop.
outlierRatioCounterMax=5 #Used for the while loop. = (outlierRatioMax - outlierRatioMin) / outlierRatioStep

outlierRatio=${outlierRatioMin}
outlierRatioCounter=${outlierRatioCounterMin} #Used for the while loop.

while [ "${outlierRatioCounter}" -ge "${outlierRatioCounterMin}" ] && [ "${outlierRatioCounter}" -le "${outlierRatioCounterMax}" ]
do
    while [ "${stdNoiseCounter}" -ge "${stdNoiseCounterMin}" ] && [ "${stdNoiseCounter}" -le "${stdNoiseCounterMax}" ]
            do

        printf "Begining for inlier noise: ${stdNoise} and outlier ratio: ${outlierRatio} .\n\n"

        genArgs="--input_model ${inputModel} --input_format ${inputFormat} --read_database ${readDatabase} --database_path ${databasePath} --noise_std ${stdNoise} --outlier_ratio ${outlierRatio} --max_try_outlier ${maxTryOutlier}"
        if [ "${delete}" -eq "1" ]; then
            genArgs="${genArgs} --delete True"
        fi
        if [ "${validate}" -eq "1" ]; then
            genArgs="${genArgs} --validate True"
        fi
        if [ "${align}" -eq "1" ]; then
            genArgs="${genArgs} --align True"
        fi

        python scripts/python/test_generate.py ${genArgs}

        expArgs="--workspace_path ${workspacePath} --image_path ${imagePath}"

        resultPath="${outputPath}/${stdNoise}_${outlierRatio}/"
        mkdir -p ${resultPath}

        build/src/exe/colmap automatic_reconstructor ${expArgs}

        mv "${expOutputPath}0/cameras.bin" "${resultPath}/ransac_cameras.bin"
        mv "${expOutputPath}0/images.bin" "${resultPath}/ransac_images.bin"
        mv "${expOutputPath}0/points3D.bin" "${resultPath}/ransac_points3D.bin"
        rm -r "${expOutputPath}0/"
        mv "TIME.txt" "${resultPath}/ransac_time.txt"

        build/src/exe/colmap_AC automatic_reconstructor ${expArgs}

        mv "${expOutputPath}0/cameras.bin" "${resultPath}/acransac_cameras.bin"
        mv "${expOutputPath}0/images.bin" "${resultPath}/acransac_images.bin"
        mv "${expOutputPath}0/points3D.bin" "${resultPath}/acransac_points3D.bin"
        rm -r "${expOutputPath}0/"
        mv "TIME.txt" "${resultPath}/acransac_time.txt"

        build/src/exe/colmap_FastAC automatic_reconstructor ${expArgs}

        mv "${expOutputPath}0/cameras.bin" "${resultPath}/fastac_cameras.bin"
        mv "${expOutputPath}0/images.bin" "${resultPath}/fastac_images.bin"
        mv "${expOutputPath}0/points3D.bin" "${resultPath}/fastac_points3D.bin"
        rm -r "${expOutputPath}0/"
        mv "TIME.txt" "${resultPath}/fastac_time.txt"

        build/src/exe/colmap_LRT automatic_reconstructor ${expArgs}

        mv "${expOutputPath}0/cameras.bin" "${resultPath}/lrt_cameras.bin"
        mv "${expOutputPath}0/images.bin" "${resultPath}/lrt_images.bin"
        mv "${expOutputPath}0/points3D.bin" "${resultPath}/lrt_points3D.bin"
        rm -r "${expOutputPath}0/"
        mv "TIME.txt" "${resultPath}/lrt_time.txt"

        mv ${databasePath} "${resultPath}/database.db"

        printf "Done for inlier noise: ${stdNoise} and outlier ratio: ${outlierRatio} .\n\n"

        stdNoise="$(echo "${stdNoise} + ${stdNoiseStep}" | bc)"
        stdNoiseCounter=$((${stdNoiseCounter}+1))
    done # end of noise std
    outlierRatio="$(echo "${outlierRatio} + ${outlierRatioStep}" | bc)"
    outlierRatioCounter=$((${outlierRatioCounter}+1))

    stdNoise=${stdNoiseMin}
    stdNoiseCounter=${stdNoiseCounterMin}

done # End of the outlier ratio loop.
