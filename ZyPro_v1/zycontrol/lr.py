from torch.optim import lr_scheduler

def zy_StepLR(optimizer,step=10,this_gamma=0.1):

    # 固定步数，乘gamma
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=this_gamma)

    return scheduler

def zy_MultiStepLR(optimizer,steps=[30, 60],this_gamma=0.1):


    #30和 60 时乘gamma

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=this_gamma)

    return scheduler

def zy_CosLR(optimizer,tmax=50,eta_min_this=0):


    #余弦退火

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min_this)

    #tmax 周期 eta_min 最小值

    return scheduler

def zy_RONP(optimizer,patience_this=5,factor_this=0.1):

    #控制和调整，达到标准调整

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_this, factor=factor_this)

    return scheduler

