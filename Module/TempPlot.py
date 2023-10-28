from plot import Plot

# copy5 50 1e-3 result5/image5 ResNet18（训 all
Plot(50, 'record/results5.json', 'images/images5',[0, 9, 24, 49])

# copy6 50 1e-3 result6/image6 ResNet18（未训 all
Plot(50, 'record/results6.json', 'images/images6', [0, 9, 24, 49])

# copy7 100 1e-3 result7/image7 ResNet18（训 all
Plot(100, 'record/results7.json', 'images/images7', [0, 9, 49, 99])

# copy8 100 1e-3 result8/image8 ResNet18（未训 all
Plot(100, 'record/results8.json', 'images/images8', [0, 9, 49, 99])

# copy9 30 1e-3 result9/image9 ResNet18（训 all
Plot(30, 'record/results9.json', 'images/images9', [0,6,14,29])

# copy10 30 1e-3 result10/image10 ResNet18（训 all
Plot(30, 'record/results10.json', 'images/images10', [0,6,14,29])



# copy15 100 1e-3 result15/image15 ResNet18（训 噪声0.2 0.01
Plot(100, 'record/results15.json', 'images/images15', [0, 9, 49, 99])

# copy16 100 1e-3 result16/image16 ResNet18（训 噪声0.2 0.0001
Plot(100, 'record/results16.json', 'images/images16', [0, 9, 49, 99])

# copy17 100 1e-3 result17/image17 ResNet18（训 噪声0.6 0.01
Plot(100, 'record/results17.json', 'images/images17', [0, 9, 49, 99])

# copy18 100 1e-3 result18/image18 ResNet18（训 噪声0.6 0.0001
Plot(100, 'record/results18.json', 'images/images18', [0, 9, 49, 99])



# copy20 100 1e-3 result18/image18 convenet（训 线性不均衡
Plot(100, 'record/results20.json', 'images/images20', [0, 9, 49, 99])

# copy21 100 1e-3 result/image21 convenet（训 [2500] * 9 + [5000]
Plot(100, 'record/results21.json', 'images/images21', [0, 9, 49, 99])

# copy22 100 1e-3 result/image22 convenet（训 [100] * 9 + [5000]
Plot(100, 'record/results22.json', 'images/images22', [0, 9, 49, 99])

# copy23 100 1e-3 result/image23 convenet（训 [500] * 5 + [5000] * 5
Plot(100, 'record/results23.json', 'images/images23', [0, 9, 49, 99])