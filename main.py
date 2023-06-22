from procimg import ProcImg


def main():
    sample_images = [
        ProcImg("maps/test1.jpg"),
        ProcImg("maps/test2.png"),
        ProcImg("maps/test3.png"),
        ProcImg("maps/test4.png"),
        ProcImg("maps/test5.jpg"),
        ProcImg("maps/test6.png"),
    ]

    for i in range(len(sample_images)):
        sample_images[i].segmentation().binarization().morph_close().skeletonization().branch_removal().plot_all_steps()


if __name__ == '__main__':
    main()
