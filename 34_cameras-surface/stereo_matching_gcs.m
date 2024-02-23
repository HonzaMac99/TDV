addpath("/home/honzamac/cvut/TDV/34_cameras-surface/")
addpath("/home/honzamac/cvut/TDV/34_cameras-surface/gcs/")

%demo for GCS matching (stereo matching via growing correspondence seeds)

load('stereo_in.mat')

Ds = cell(1, length(task));

for i = 1:length(task)
    img1 = task{i, 1};
    img2 = task{i, 2};
    seeds = task{i, 3};


    % % Display the first image in the first subplot
    % pause(3);
    % imshow(img1);
    % title('Image 1');
    % drawnow
    
    % % Display the second image in the second subplot
    % imshow(img2);
    % title('Image 2');

    % gcs gives negative disparity fields

    % figure
    Ds{i} = - gcs(img1, img2, seeds);

    imagesc(Ds{i}); axis image; colormap(jet); colorbar;
    title('Disparity map');
    drawnow
end

save("stereo_out.mat", "Ds")



