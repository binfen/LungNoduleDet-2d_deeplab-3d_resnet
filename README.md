## 文件夹说明

    [1] segment
        分割阶段的过程

    [2] classify
        分类阶段的过程

    [3] evalutation
        整体模型效果评价

    [4] demo-locate
        本地 demo

    [5] demo-service
        服务端 demo
    
    [6] preprocess
        制作分割，分类阶段的训练集，验证集，测试集


## demo-locate
    '''
    python ../demo-locate/main.py
    '''

    [note:] 具体参数设置查看 ‘../segment/opts.py’



## demo-service
    '''
    python ../demo-service/main.py
    '''

    [note:] 具体参数设置查看 ‘../segment/opts.py’



## training：

    [step1:训练分割] 
        '''
        python ../segment/main.py
        '''

        [note:] 具体参数设置查看 ‘../segment/opts.py’

    [step2:分类训练]
        '''
        python ../classify/main.py
        '''

        [note:] 具体参数设置查看 ‘../segment/opts.py’

## evalation：
    '''
    python  ../evalutation/predict.py
    '''

    [note:] 具体参数设置查看 ‘../segment/opts.py’
