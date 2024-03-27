IMAGE_MONO = '/home/david/data/Husky_Bags/lab/2022-07-04-10-28-32/log/left_kf.monolithic'
EXP_MONO = '/home/david/data/Husky_Bags/New_lab/2022-08-31-05-01-56_exp.monolithic'
KF_EXP_MONO = '/home/david/data/Husky_Bags/New_lab/2022-08-31-05-01-56_keyframe_exp.monolithic'
import aru_py_logger
import cv2


def view_exp_monolithic(exp_mono, wait_ms = 0):
    global i, image
    exp_logger = aru_py_logger.ExperienceLogger(exp_mono, False)
    i = 0
    while not exp_logger.end_of_file():
        _, image, *_ = exp_logger.read_from_file()
        # print(exp_logger.read_from_file())
        cv2.imshow('Image', image)
        cv2.resizeWindow('Image', width=int(image.shape[1] * 0.4), height=int(image.shape[0] * 0.4))
        cv2.waitKey(wait_ms)
        i += 1
    print(f'{i} messages read.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_logger = aru_py_logger.MonoImageLogger(IMAGE_MONO, False)
    #
    # while not image_logger.end_of_file():
    #     image, _ = image_logger.read_from_file()
    #     cv2.imshow('Image', image)
    #     cv2.resizeWindow('Image', width=int(image.shape[1] * 0.4), height=int(image.shape[0] * 0.4))
    #     cv2.waitKey(100)
    view_exp_monolithic(EXP_MONO, wait_ms=10)
    view_exp_monolithic(KF_EXP_MONO, wait_ms=100)