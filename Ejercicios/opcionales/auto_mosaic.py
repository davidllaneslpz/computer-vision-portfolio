#!/usr/bin/env python3
"""Crea un mosaico automático desde una carpeta de imágenes usando homografías.
Compara el resultado con el stitching de OpenCV.
"""

import argparse
import glob
import os
import sys

import cv2 as cv
import numpy as np


def list_images(folder):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    images = []
    for ext in exts:
        images.extend(sorted(glob.glob(os.path.join(folder, ext))))
    return images


def create_detector():
    """Return feature detector: SIFT if available, else ORB fallback."""
    if hasattr(cv, 'SIFT_create'):
        return cv.SIFT_create(), 'SIFT'
    if hasattr(cv, 'xfeatures2d') and hasattr(cv.xfeatures2d, 'SIFT_create'):
        return cv.xfeatures2d.SIFT_create(), 'SIFT'
    return cv.ORB_create(5000), 'ORB'


def match_features(desc1, desc2, matcher, use_sift):
    """Match descriptors using Lowe's ratio test."""
    if desc1 is None or desc2 is None:
        return []
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    ratio_thresh = 0.75 if use_sift else 0.85
    for m, n in knn:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def compute_homography(kp1, desc1, kp2, desc2, matcher, use_sift, min_matches=12):
    matches = match_features(desc1, desc2, matcher, use_sift)
    if len(matches) < min_matches:
        return None, []
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
    if H is None or mask is None:
        return None, []
    inliers = [m for m, v in zip(matches, mask.ravel()) if v]
    if len(inliers) < min_matches:
        return None, inliers
    return H, inliers


def build_pairwise_graph(images, kps, descs, matcher, use_sift):
    n = len(images)
    graph = {i: [] for i in range(n)}
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            H, inliers = compute_homography(kps[i], descs[i], kps[j], descs[j], matcher, use_sift)
            if H is None or len(inliers) < 12:
                continue
            inlier_count = len(inliers)
            graph[i].append((j, H, inlier_count))
            graph[j].append((i, np.linalg.inv(H), inlier_count))
            scores.append((inlier_count, i, j))
    return graph, scores


def largest_connected_component(graph):
    """Find largest connected component using DFS."""
    visited = set()
    components = []
    for start in graph:
        if start in visited:
            continue
        stack = [start]
        comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for neigh, _, _ in graph[node]:
                if neigh not in visited:
                    stack.append(neigh)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return components[0] if components else []


def choose_reference(graph):
    """Choose central image with highest connection strength."""
    best = None
    best_degree = -1
    for i, neighbors in graph.items():
        degree = sum(score for _, _, score in neighbors)
        if degree > best_degree:
            best_degree = degree
            best = i
    return best


def compute_global_transforms(graph, reference):
    """Propagate reference frame transforms to all other images via BFS."""
    transforms = {reference: np.eye(3, dtype=np.float64)}
    queue = [reference]
    visited = {reference}
    while queue:
        i = queue.pop(0)
        H_i = transforms[i]
        for j, H_j_to_i, _ in graph[i]:
            if j in visited:
                continue
            transforms[j] = H_i @ H_j_to_i
            visited.add(j)
            queue.append(j)
    return transforms


def warp_image_to_canvas(img, H, size):
    return cv.warpPerspective(img, H, size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)


def create_mosaic(images, transforms):
    """Blend warped images into single canvas with averaging."""
    corners = []
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv.perspectiveTransform(pts, H)
        corners.append(warped.reshape(-1, 2))
    all_pts = np.vstack(corners)
    min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)
    shift = np.array([-min_x, -min_y], dtype=float)
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError('Canvas size invalid: %dx%d' % (canvas_w, canvas_h))

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    offset = np.eye(3, dtype=np.float64)
    offset[0, 2] = shift[0]
    offset[1, 2] = shift[1]

    for img, H in zip(images, transforms):
        H_off = offset @ H
        warped = warp_image_to_canvas(img, H_off, (canvas_w, canvas_h)).astype(np.float32)
        mask = warp_image_to_canvas(np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255, H_off, (canvas_w, canvas_h))
        mask_bool = (mask[:, :, 0] if mask.ndim == 3 else mask) > 0
        canvas[mask_bool] += warped[mask_bool]
        weight[mask_bool] += 1.0

    normalized = np.zeros_like(canvas, dtype=np.uint8)
    nonzero = weight > 0
    normalized[nonzero] = np.clip(canvas[nonzero] / weight[nonzero, None], 0, 255).astype(np.uint8)
    return normalized


def compare_opencv_stitcher(img_list, output_path):
    if len(img_list) < 2:
        raise ValueError('Need at least two images for OpenCV stitching')
    stitcher = cv.Stitcher_create(cv.Stitcher_SCANS)
    status, pano = stitcher.stitch(img_list)
    if status != cv.Stitcher_OK:
        stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(img_list)
    if status != cv.Stitcher_OK:
        raise RuntimeError('OpenCV stitcher failed with code %d' % status)
    cv.imwrite(output_path, pano)
    return pano


def summarize_results(images, graph, connected, reference, transforms, output, opencv_output):
    print('Images loaded:', len(images))
    print('Connected component size:', len(connected))
    print('Reference image index:', reference)
    print('Edges used:')
    for i, neighbors in graph.items():
        for j, _, score in neighbors:
            if i < j:
                print('  %d <-> %d  inliers=%d' % (i, j, score))
    print('Mosaic saved to', output)
    print('OpenCV stitch result saved to', opencv_output)


def main():
    parser = argparse.ArgumentParser(description='Auto mosaic from folder using homographies.')
    parser.add_argument('--dir', required=True, help='Folder with input images')
    parser.add_argument('--output', default='auto_mosaic.jpg', help='Output mosaic image')
    parser.add_argument('--opencv-output', default='opencv_mosaic.jpg', help='OpenCV stitching result')
    parser.add_argument('--show', action='store_true', help='Show intermediate and final images')
    parser.add_argument('--min-matches', type=int, default=12, help='Minimum feature matches for homography')
    args = parser.parse_args()

    image_paths = list_images(args.dir)
    if not image_paths:
        print('No images found in', args.dir)
        return

    print('Found', len(image_paths), 'images')
    images = [cv.imread(p) for p in image_paths]
    valid = [i for i, img in enumerate(images) if img is not None]
    if len(valid) != len(images):
        print('Warning: some images could not be read and will be skipped')
    images = [images[i] for i in valid]
    image_paths = [image_paths[i] for i in valid]

    detector, det_name = create_detector()
    if det_name == 'SIFT':
        matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        use_sift = True
    else:
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        use_sift = False

    keypoints = []
    descriptors = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(desc)

    graph, scores = build_pairwise_graph(image_paths, keypoints, descriptors, matcher, use_sift)
    if not any(graph.values()):
        raise RuntimeError('No valid homographies found between image pairs')

    component = sorted(largest_connected_component(graph))
    if len(component) < len(images):
        print('Warning: only', len(component), 'images are connected; skipping the rest')
        old_graph = graph
        remap = {old: new for new, old in enumerate(component)}
        images = [images[i] for i in component]
        keypoints = [keypoints[i] for i in component]
        descriptors = [descriptors[i] for i in component]
        image_paths = [image_paths[i] for i in component]
        graph = {}
        for new_i, old_i in enumerate(component):
            graph[new_i] = []
            for old_j, H, s in old_graph[old_i]:
                if old_j in remap:
                    graph[new_i].append((remap[old_j], H, s))

    reference = choose_reference(graph)
    transforms = compute_global_transforms(graph, reference)
    ordered_transforms = [transforms[i] for i in range(len(images))]

    mosaic = create_mosaic(images, ordered_transforms)
    cv.imwrite(args.output, mosaic)

    try:
        opencv_mosaic = compare_opencv_stitcher(images, args.opencv_output)
    except Exception as e:
        print('OpenCV stitching failed:', e)
        opencv_mosaic = None

    summarize_results(images, graph, component, reference, transforms, args.output, args.opencv_output)

    if args.show:
        cv.imshow('Auto mosaic', mosaic)
        if opencv_mosaic is not None:
            cv.imshow('OpenCV stitch', opencv_mosaic)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
