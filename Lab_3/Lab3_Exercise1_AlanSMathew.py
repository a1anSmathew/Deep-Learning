def compute_f2(f1,z):
    return f1*z

def compute_f1(x,y):
    return x+y

def forward_pass(x,y,z):
    f1 = compute_f1(x,y)
    f2 = compute_f2(f1,z)

    return f1,f2

def compute_df2_df2():
    return 1

def compute_df2_df1(z):
    return z

def compute_df2_z(f1):
    return f1

def compute_df2_x(f1):
    g = f1
    l = 1
    return g * l

def compute_df2_y(f1):
    g = f1
    l = 1
    return g * l


def backprop(x,y,z):
    f1,f2 = forward_pass(x,y,z)
    df2_f2 = compute_df2_df2()
    df2_f1 = compute_df2_df1(z)
    global_grad = df2_f1
    df2_z = compute_df2_z(f1)
    df2_x = compute_df2_x(df2_f1)
    df2_y = compute_df2_y(df2_f1)

    return f1,f2,df2_f2,df2_f1,df2_z,df2_x,df2_y



def main():
    x,y,z = -2,5,-4
    f1,f2,df2_f2,df2_f1,df2_z,df2_x,df2_y = backprop(x,y,z)
    print(f"f1 = {f1}\n"
          f"f2 = {f2}\n"
          f"df2/df2 = {df2_f2}\n"
          f"df2/df1 = {df2_f1}\n"
          f"df2/dz = {df2_z}\n"
          f"df2/dx = {df2_x}\n"
          f"df2/dy = {df2_y}\n")

if __name__ == '__main__':
    main()