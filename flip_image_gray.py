def mirror_this(image_file, gray_scale=False, with_plot=False):
    image_rgb = read_this(image_file=image_file, gray_scale=gray_scale)
    image_mirror = np.fliplr(image_rgb)
    if with_plot:
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Mirrored")
        if not gray_scale:
            ax1.imshow(image_rgb)
            ax2.imshow(image_mirror)
        else:
            ax1.imshow(image_rgb, cmap='gray')
            ax2.imshow(image_mirror, cmap='gray')
        return True
    return image_mirror
mirror_this(image_file="lena_original.png", gray_scale=True, with_plot=True)