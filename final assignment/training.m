function training(str, features, output)
    % x are the PCA features.
    % t is the target data.
    x = features';
    t = output';
    % Training using levelberg-Marquardt backpropagation
    trainFcn = str;
    hiddenLayerSize = [6 12];
    net = fitnet(hiddenLayerSize,trainFcn);
    % Dividing the data 
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 20/100;
    net.trainParam.showWindow = false;
    [net,tr] = train(net,x,t);
    y = net(x);
    figure, plotperform(tr)
    figure, plotregression(t,y)
return
