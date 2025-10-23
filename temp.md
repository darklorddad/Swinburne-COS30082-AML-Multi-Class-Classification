INFO     | 2025-10-23 16:39:02 | autotrain.app.ui_routes:handle_form:550 - hardware: local-ui
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4813/4813 [00:00<00:00, 32781.46it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1220/1220 [00:00<00:00, 24141.25it/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4813/4813 [00:00<00:00, 38408.32files/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1220/1220 [00:00<00:00, 50428.70files/s]
Generating train split: 4813 examples [00:00, 21896.42 examples/s]
Generating validation split: 1220 examples [00:00, 19271.30 examples/s]
Saving the dataset (2/2 shards): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4813/4813 [00:09<00:00, 483.80 examples/s]
Saving the dataset (1/1 shards): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1220/1220 [00:02<00:00, 598.70 examples/s]
INFO     | 2025-10-23 16:40:11 | autotrain.backends.local:create:20 - Starting local training...
INFO     | 2025-10-23 16:40:11 | autotrain.commands:launch_command:514 - ['accelerate', 'launch', '--cpu', '-m', 'autotrain.trainers.image_classification', '--training_config', 'Model-SwinV2-Tiny-\\training_params.json']
INFO     | 2025-10-23 16:40:11 | autotrain.commands:launch_command:515 - {'data_path': 'Model-SwinV2-Tiny-/autotrain-data', 'model': 'microsoft/swinv2-tiny-patch4-window8-256', 'username': 'local', 'lr': 5e-05, 'epochs': 100, 'batch_size': 32, 'warmup_ratio': 0.1, 'gradient_accumulation': 2, 'optimizer': 'adamw_torch', 'scheduler': 'cosine_warmup', 'weight_decay': 0.01, 'max_grad_norm': 1.0, 'seed': 42, 'train_split': 'train', 'valid_split': 'validation', 'logging_steps': -1, 'project_name': 'Model-SwinV2-Tiny-', 'auto_find_batch_size': False, 'mixed_precision': 'fp16', 'save_total_limit': 1, 'token': None, 'push_to_hub': True, 'eval_strategy': 'epoch', 'image_column': 'autotrain_image', 'target_column': 'autotrain_label', 'log': 'tensorboard', 'early_stopping_patience': 5, 'early_stopping_threshold': 0.01}
INFO     | 2025-10-23 16:40:11 | autotrain.backends.local:create:25 - Training PID: 8608
INFO:     127.0.0.1:53924 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54031 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54185 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54190 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54191 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:53924 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54031 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54185 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54190 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54191 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:53924 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54031 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54185 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54190 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54191 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54031 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:53924 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54191 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54190 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:53924 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:53902 - "POST /ui/create_project HTTP/1.1" 200 OK
INFO:     127.0.0.1:54190 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:53902 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54190 - "GET /ui/accelerators HTTP/1.1" 200 OK
The following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_processes` was set to a value of `0`
        `--num_machines` was set to a value of `1`
        `--mixed_precision` was set to a value of `'no'`
        `--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
INFO:     127.0.0.1:54228 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54228 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54228 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54228 - "GET /ui/is_model_training HTTP/1.1" 200 OK
C:\Users\102777801\AppData\Roaming\Python\Python312\site-packages\albumentations\__init__.py:24: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.23). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
INFO     | 2025-10-23 16:40:33 | __main__:train:86 - Train data: Dataset({
    features: ['autotrain_image', 'autotrain_label'],
    num_rows: 4813
})
INFO     | 2025-10-23 16:40:33 | __main__:train:87 - Valid data: Dataset({
    features: ['autotrain_image', 'autotrain_label'],
    num_rows: 1220
})
INFO     | 2025-10-23 16:40:33 | __main__:train:90 - Classes: ['acadian_flycatcher', 'american_crow', 'american_goldfinch', 'american_pipit', 'american_redstart', 'american_three_toed_woodpecker', 'anna_hummingbird', 'artic_tern', 'baird_sparrow', 'baltimore_oriole', 'bank_swallow', 'barn_swallow', 'bay_breasted_warbler', 'belted_kingfisher', 'bewick_wren', 'black_and_white_warbler', 'black_billed_cuckoo', 'black_capped_vireo', 'black_footed_albatross', 'black_tern', 'black_throated_blue_warbler', 'black_throated_sparrow', 'blue_grosbeak', 'blue_headed_vireo', 'blue_jay', 'blue_winged_warbler', 'boat_tailed_grackle', 'bobolink', 'bohemian_waxwing', 'brandt_cormorant', 'brewer_blackbird', 'brewer_sparrow', 'bronzed_cowbird', 'brown_creeper', 'brown_pelican', 'brown_thrasher', 'cactus_wren', 'california_gull', 'canada_warbler', 'cape_glossy_starling', 'cape_may_warbler', 'cardinal', 'carolina_wren', 'caspian_tern', 'cedar_waxwing', 'cerulean_warbler', 'chestnut_sided_warbler', 'chipping_sparrow', 'chuck_will_widow', 'clark_nutcracker', 'clay_colored_sparrow', 'cliff_swallow', 'common_raven', 'common_tern', 'common_yellowthroat', 'crested_auklet', 'dark_eyed_junco', 'downy_woodpecker', 'eared_grebe', 'eastern_towhee', 'elegant_tern', 'european_goldfinch', 'evening_grosbeak', 'field_sparrow', 'fish_crow', 'florida_jay', 'forsters_tern', 'fox_sparrow', 'frigatebird', 'gadwall', 'geococcyx', 'glaucous_winged_gull', 'golden_winged_warbler', 'grasshopper_sparrow', 'gray_catbird', 'gray_crowned_rosy_finch', 'gray_kingbird', 'great_crested_flycatcher', 'great_grey_shrike', 'green_jay', 'green_kingfisher', 'green_tailed_towhee', 'green_violetear', 'groove_billed_ani', 'harris_sparrow', 'heermann_gull', 'henslow_sparrow', 'herring_gull', 'hooded_merganser', 'hooded_oriole', 'hooded_warbler', 'horned_grebe', 'horned_lark', 'horned_puffin', 'house_sparrow', 'house_wren', 'indigo_bunting', 'ivory_gull', 'kentucky_warbler', 'laysan_albatross', 'lazuli_bunting', 'le_conte_sparrow', 'least_auklet', 'least_flycatcher', 'least_tern', 'lincoln_sparrow', 'loggerhead_shrike', 'long_tailed_jaeger', 'louisiana_waterthrush', 'magnolia_warbler', 'mallard', 'mangrove_cuckoo', 'marsh_wren', 'mockingbird', 'mourning_warbler', 'myrtle_warbler', 'nashville_warbler', 'nelson_sharp_tailed_sparrow', 'nighthawk', 'northern_flicker', 'northern_fulmar', 'northern_waterthrush', 'olive_sided_flycatcher', 'orange_crowned_warbler', 'orchard_oriole', 'ovenbird', 'pacific_loon', 'painted_bunting', 'palm_warbler', 'parakeet_auklet', 'pelagic_cormorant', 'philadelphia_vireo', 'pied_billed_grebe', 'pied_kingfisher', 'pigeon_guillemot', 'pileated_woodpecker', 'pine_grosbeak', 'pine_warbler', 'pomarine_jaeger', 'prairie_warbler', 'prothonotary_warbler', 'purple_finch', 'red_bellied_woodpecker', 'red_breasted_merganser', 'red_cockaded_woodpecker', 'red_eyed_vireo', 'red_faced_cormorant', 'red_headed_woodpecker', 'red_legged_kittiwake', 'red_winged_blackbird', 'rhinoceros_auklet', 'ring_billed_gull', 'ringed_kingfisher', 'rock_wren', 'rose_breasted_grosbeak', 'ruby_throated_hummingbird', 'rufous_hummingbird', 'rusty_blackbird', 'sage_thrasher', 'savannah_sparrow', 'sayornis', 'scarlet_tanager', 'scissor_tailed_flycatcher', 'scott_oriole', 'seaside_sparrow', 'shiny_cowbird', 'slaty_backed_gull', 'song_sparrow', 'sooty_albatross', 'spotted_catbird', 'summer_tanager', 'swainson_warbler', 'tennessee_warbler', 'tree_sparrow', 'tree_swallow', 'tropical_kingbird', 'vermilion_flycatcher', 'vesper_sparrow', 'warbling_vireo', 'western_grebe', 'western_gull', 'western_meadowlark', 'western_wood_pewee', 'whip_poor_will', 'white_breasted_kingfisher', 'white_breasted_nuthatch', 'white_crowned_sparrow', 'white_eyed_vireo', 'white_necked_raven', 'white_pelican', 'white_throated_sparrow', 'wilson_warbler', 'winter_wren', 'worm_eating_warbler', 'yellow_bellied_flycatcher', 'yellow_billed_cuckoo', 'yellow_breasted_chat', 'yellow_headed_blackbird', 'yellow_throated_vireo', 'yellow_warbler']
INFO:     127.0.0.1:54229 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54229 - "GET /ui/is_model_training HTTP/1.1" 200 OK
config.json: 69.9kB [00:00, ?B/s]
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
pytorch_model.bin:  46%|██████████████████████████████████████████████████████████████████████████████████████████                                                                                                         | 52.4M/113M [00:02<00:02, 21.8MB/s]INFO:     127.0.0.1:54229 - "GET /ui/is_model_training HTTP/1.1" 200 OK
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 113M/113M [00:05<00:00, 22.3MB/s]
Some weights of Swinv2ForImageClassification were not initialized from the model checkpoint at microsoft/swinv2-tiny-patch4-window8-256 and are newly initialized because the shapes did not match:
- classifier.weight: found shape torch.Size([1000, 768]) in the checkpoint and torch.Size([200, 768]) in the model instantiated
- classifier.bias: found shape torch.Size([1000]) in the checkpoint and torch.Size([200]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
preprocessor_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:00<?, ?B/s]
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.48, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
INFO     | 2025-10-23 16:40:43 | __main__:train:152 - Logging steps: 7
INFO     | 2025-10-23 16:40:43 | autotrain.trainers.common:on_train_begin:386 - Starting to train...
  0%|                                                                                                                                                                                                                                 | 0/7500 [00:00<?, ?it/s]C:\Users\102777801\AppData\Roaming\Python\Python312\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
INFO:     127.0.0.1:54229 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54229 - "GET /ui/is_model_training HTTP/1.1" 200 OK
                                                                                                                                                                                                                                                               INFO:     127.0.0.1:54249 - "GET /ui/is_model_training HTTP/1.1" 200 OK██████████████████████████████████████████████████████████████▏                                                                                      | 62.9M/113M [00:04<00:03, 13.2MB/s]
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 113M/113M [00:07<00:00, 15.7MB/s]
INFO:     127.0.0.1:54249 - "GET /ui/accelerators HTTP/1.1" 200 OK██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 113M/113M [00:07<00:00, 19.4MB/s]
INFO:     127.0.0.1:54249 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54253 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54253 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54253 - "GET /ui/is_model_training HTTP/1.1" 200 OK
  0%|                                                                                                                                                                                                                      | 1/7500 [00:31<64:39:00, 31.04s/it]INFO:     127.0.0.1:54260 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54260 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54260 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54285 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54290 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54290 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:54291 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:54291 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:57659 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:57660 - "GET /ui/is_model_training HTTP/1.1" 200 OK
  0%|                                                                                                                                                                                                                      | 2/7500 [01:11<76:14:40, 36.61s/it]INFO:     127.0.0.1:57660 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:57660 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:57662 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62422 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62422 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62423 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62423 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62431 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62432 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62432 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62432 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62433 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62433 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62433 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62433 - "GET /ui/is_model_training HTTP/1.1" 200 OK
  0%|                                                                                                                                                                                                                      | 3/7500 [02:04<92:11:36, 44.27s/it]INFO:     127.0.0.1:62433 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62446 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62446 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62446 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62452 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62452 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62452 - "GET /ui/is_model_training HTTP/1.1" 200 OK
  0%|                                                                                                                                                                                                                      | 4/7500 [02:54<96:22:46, 46.29s/it]INFO:     127.0.0.1:62458 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62458 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62458 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62458 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62458 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62458 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62490 - "GET /ui/is_model_training HTTP/1.1" 200 OK
  0%|▏                                                                                                                                                                                                                     | 5/7500 [03:25<85:00:15, 40.83s/it]INFO:     127.0.0.1:62490 - "GET /ui/is_model_training HTTP/1.1" 200 OK
INFO:     127.0.0.1:62490 - "GET /ui/accelerators HTTP/1.1" 200 OK
INFO:     127.0.0.1:62490 - "GET /ui/is_model_training HTTP/1.1" 200 OK
  0%|▏                                                                                                                                                                                                                     | 7/7500 [04:22<70:50:06, 34.03s/it]INFO     | 2025-10-23 16:45:06 | autotrain.trainers.common:on_log:367 - {'loss': 5.3102, 'grad_norm': 3.0164265632629395, 'learning_rate': 4.666666666666667e-07, 'epoch': 0.09271523178807947}
{'loss': 5.3102, 'grad_norm': 3.0164265632629395, 'learning_rate': 4.666666666666667e-07, 'epoch': 0.09}
  0%|▎                                                                                                                                                                                                                    | 11/7500 [06:23<64:56:49, 31.22s/it]