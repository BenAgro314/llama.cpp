#include "ggml.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"

#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <nlohmann/json.hpp>
#include <fstream>

void save_data_to_file(const std::string &out_path, std::string name, std::string video_path, double fps, std::vector<int> frames, std::vector<std::string> captions)
{
    nlohmann::json data;
    data["name"] = name;
    data["video_path"] = video_path;
    data["fps"] = fps;
    data["frames"] = frames;
    data["captions"] = captions;

    std::ofstream file(out_path + "/" + name + ".json");
    if (file.is_open())
    {
        file << data.dump(4);
        file.close();
    }
    else
    {
        std::cerr << "Could not open file for writing." << std::endl;
    }
}

std::map<std::string, std::string> read_video_meatdata(const std::string &file_name)
{
    std::ifstream file(file_name);
    nlohmann::json j;
    file >> j;

    std::map<std::string, std::string> data_map;

    for (auto &element : j.items())
    {
        std::string key = element.key();
        std::string value;

        if (element.value().is_string())
        {
            value = element.value();
        }
        else
        {
            value = element.value().dump(); // Convert non-string value to a JSON string
        }

        data_map[key] = value;
    }

    return data_map;
}

std::pair<std::vector<std::pair<int, int>>, std::vector<std::string>> read_audio_captions(const std::string &filename)
{
    // Read the JSON file
    std::ifstream ifs(filename);
    nlohmann::json j;
    ifs >> j;

    // Vectors to store the timestamps and transcriptions
    std::vector<std::pair<int, int>> timestamps_ms;
    std::vector<std::string> transcriptions;

    // Iterate over the transcription array
    for (const auto &item : j["transcription"])
    {
        int start = item["offsets"]["from"];
        int end = item["offsets"]["to"];
        std::string text = item["text"];

        // Add the data to the vectors
        timestamps_ms.push_back(std::make_pair(start, end));
        transcriptions.push_back(text);
    }

    return std::make_pair(timestamps_ms, transcriptions);
}

std::pair<std::string, std::string> split_ext(const std::string &filename)
{
    size_t last_dot = filename.find_last_of('.');

    if (last_dot != std::string::npos && last_dot != 0)
    {
        return {filename.substr(0, last_dot), filename.substr(last_dot)};
    }
    else
    {
        return {filename, ""}; // No extension found or dot is at the beginning
    }
}

bool file_exists(const std::string &path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::pair<std::string, std::string> split_path(const std::string &path)
{
    size_t found = path.find_last_of("/\\");
    if (found != std::string::npos)
    {
        return {path.substr(0, found), path.substr(found + 1)};
    }
    return {"", path}; // In case there is no '/' or '\' in the path
}

std::vector<std::string> list_files_in_directory(const std::string &directory_path)
{
    std::vector<std::string> file_paths;
    DIR *dir;
    struct dirent *ent;
    struct stat st;

    dir = opendir(directory_path.c_str());
    if (dir != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            const std::string file_name = ent->d_name;
            const std::string full_file_name = directory_path + "/" + file_name;

            if (file_name[0] == '.')
                continue; // Skip hidden files

            if (stat(full_file_name.c_str(), &st) == -1)
                continue; // Handle error

            const bool is_directory = (st.st_mode & S_IFDIR) != 0;

            if (is_directory)
                continue; // Skip directories

            file_paths.push_back(full_file_name);
        }
        closedir(dir);
    }
    else
    {
        perror("opendir");
    }

    return file_paths;
}
static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens, int n_batch, int *n_past)
{
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch)
    {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
        {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0)))
        {
            fprintf(stderr, "%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past)
{
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context *ctx_llama, const char *str, int n_batch, int *n_past, bool add_bos)
{
    std::string str2 = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

// TODO: use common/sampling.h
static llama_token sample_id(llama_context *ctx_llama, gpt_params &params)
{
    auto &sparams = params.sparams;

    // out of user input, sample next token
    const float temp = sparams.temp;
    const int32_t top_k = sparams.top_k <= 0 ? llama_n_vocab(llama_get_model(ctx_llama)) : sparams.top_k;
    const float top_p = sparams.top_p;
    const float tfs_z = sparams.tfs_z;
    const float typical_p = sparams.typical_p;
    // const int32_t repeat_last_n   = sparams.repeat_last_n < 0 ? n_ctx : sparams.repeat_last_n;
    // const float   repeat_penalty  = sparams.repeat_penalty;
    // const float   alpha_presence  = sparams.presence_penalty;
    // const float   alpha_frequency = sparams.frequency_penalty;
    const int mirostat = sparams.mirostat;
    const float mirostat_tau = sparams.mirostat_tau;
    const float mirostat_eta = sparams.mirostat_eta;
    // const bool    penalize_nl     = sparams.penalize_nl;

    llama_token id = 0;
    {
        auto logits = llama_get_logits(ctx_llama);
        auto n_vocab = llama_n_vocab(llama_get_model(ctx_llama));

        // Apply params.logit_bias map
        for (auto it = sparams.logit_bias.begin(); it != sparams.logit_bias.end(); it++)
        {
            logits[it->first] += it->second;
        }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        if (temp <= 0)
        {
            // Greedy sampling
            id = llama_sample_token_greedy(ctx_llama, &candidates_p);
        }
        else
        {
            if (mirostat == 1)
            {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token_mirostat(ctx_llama, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            }
            else if (mirostat == 2)
            {
                static float mirostat_mu = 2.0f * mirostat_tau;
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token_mirostat_v2(ctx_llama, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            }
            else
            {
                // Temperature sampling
                llama_sample_top_k(ctx_llama, &candidates_p, top_k, 1);
                llama_sample_tail_free(ctx_llama, &candidates_p, tfs_z, 1);
                llama_sample_typical(ctx_llama, &candidates_p, typical_p, 1);
                llama_sample_top_p(ctx_llama, &candidates_p, top_p, 1);
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token(ctx_llama, &candidates_p);
            }
        }
    }

    return id;
}

static const char *sample(struct llama_context *ctx_llama, gpt_params &params, int *n_past)
{
    int id = sample_id(ctx_llama, params);
    static std::string ret;
    if (id == llama_token_eos(llama_get_model(ctx_llama)))
    {
        ret = "</s>";
    }
    else
    {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

struct llava_context
{
    struct clip_ctx *ctx_clip = NULL;
    struct llama_context *ctx_llama = NULL;
    struct llama_model *model = NULL;
};

static void show_additional_info(int /*argc*/, char **argv)
{
    printf("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    printf("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed *load_image_embed(llava_context *ctx_llava, gpt_params *params, std::string image_path)
{

    // load and preprocess the image
    llava_image_embed *embed = NULL;
    embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->n_threads, image_path.c_str());
    if (!embed)
    {
        fprintf(stderr, "%s: is %s really an image file?\n", __func__, image_path.c_str());
        return NULL;
    }

    return embed;
}

static std::string process_prompt(
    struct llava_context *ctx_llava,
    struct llava_image_embed *image_embed,
    gpt_params *params,
    const std::string &prompt,
    std::map<std::string, std::string> video_metadata,
    std::string audio_caption)
{
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    // llava chat format is "<system_prompt>\nUSER:<image_embeddings>\n<textual_prompt>\nASSISTANT:"
    std::string sys_prompt = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.";
    eval_string(ctx_llava->ctx_llama, (sys_prompt + "\nUSER:I am watching a video.").c_str(), params->n_batch, &n_past, true);
    if (video_metadata.size() > 0)
    {
        std::string metadata_prompt = "The video has the following metadata:\n";
        // prompt_with_metadata += "The video has the following metadata:\n";
        for (const auto &[key, value] : video_metadata)
        {
            metadata_prompt += key + ": " + value + "\n";
        }
        eval_string(ctx_llava->ctx_llama, (metadata_prompt).c_str(), params->n_batch, &n_past, true);
    }
    eval_string(ctx_llava->ctx_llama, "The video is at the following image:", params->n_batch, &n_past, true);
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    if (audio_caption.size() > 0)
    {
        std::string audio_prompt = "With the following closed captions: " + audio_caption + "\n";
        eval_string(ctx_llava->ctx_llama, (audio_caption).c_str(), params->n_batch, &n_past, true);
    }
    if ((audio_caption.size() > 0) || (video_metadata.size() > 0))
    {
        eval_string(ctx_llava->ctx_llama, "Using this context, caption the image in detail\nASSISTANT:", params->n_batch, &n_past, false);
    }
    else
    {
        eval_string(ctx_llava->ctx_llama, "Caption the image in detail\nASSISTANT:", params->n_batch, &n_past, false);
    }
    std::string result;
    for (int i = 0; i < max_tgt_len; i++)
    {
        const char *tmp = sample(ctx_llava->ctx_llama, *params, &n_past);
        if (strcmp(tmp, "</s>") == 0)
            break;
        result += tmp;
    }
    return result;
}

static struct llava_context *llava_init(gpt_params *params)
{
    const char *clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty())
    {
        prompt = "The image comes from a video. Caption this image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/1);

    llama_backend_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model *model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return NULL;
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context *ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL)
    {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context *ctx_llava)
{
    if (ctx_llava->ctx_clip)
    {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

int main(int argc, char **argv)
{
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params))
    {
        show_additional_info(argc, argv);
        return 1;
    }
    if (params.mmproj.empty() || params.video_dir.empty())
    {
        gpt_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }

    auto ctx_llava = llava_init(&params);
    if (ctx_llava == NULL)
    {
        fprintf(stderr, "%s: error: failed to init llava\n", __func__);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx_llava->ctx_llama);

    std::vector<std::string> video_paths = list_files_in_directory(params.video_dir);

    // iterate over videos to process
    for (auto video_path : video_paths)
    {
        auto [parent, stem] = split_path(video_path);
        auto [file_name, ext] = split_ext(stem);

        auto doc_path = params.doc_dir + "/" + file_name + ".json";
        auto metadata_path = params.video_metadata_dir + "/" + file_name + ".json";
        auto audio_captions_path = params.audio_captions_dir + "/" + file_name + ".json";
        if (file_exists(doc_path))
        {
            std::cout << "Continuing because document " << doc_path << " exists" << std::endl;
            continue;
        }

        std::map<std::string, std::string> video_metadata;
        if (file_exists(metadata_path))
        {
            video_metadata = read_video_meatdata(metadata_path);
        }

        std::pair<std::vector<std::pair<int, int>>, std::vector<std::string>> audio_captions;
        if (file_exists(audio_captions_path))
        {
            audio_captions = read_audio_captions(audio_captions_path);
        }
        bool audio_captions_exist = !audio_captions.first.empty() || !audio_captions.second.empty();

        int count = 0;
        long out_fps = 1; // process one frame every second for my sanity
        cv::VideoCapture cap(video_path);
        double fps = cap.get(cv::CAP_PROP_FPS);
        cap.release();
        auto frame_bytes_list = read_video_frames_to_bytes(video_path, out_fps);
        auto num_frames = frame_bytes_list.size();
        std::cout << "Processing " << num_frames << " frames from video " << video_path << std::endl;
        std::vector<std::string> captions;
        std::vector<int> frame_inds;
        for (auto [frame_bytes, frame_ind] : frame_bytes_list)
        {
            long image_bytes_length = frame_bytes.size(); // Get the size of the data
            const unsigned char *image_bytes = frame_bytes.data();
            auto image_embed = llava_image_embed_make_with_bytes(ctx_llava->ctx_clip, params.n_threads, image_bytes, image_bytes_length);
            std::string audio_caption = "";
            if (audio_captions_exist)
            {
                // find corresponding caption during this frame, if it exists
                float t_ms = ((1 / fps) * frame_ind) * 1000.0;
                int caption_count = 0;
                for (std::pair<int, int> timestamp_ms : audio_captions.first)
                {
                    if (t_ms >= (float)timestamp_ms.first && (t_ms <= (float)timestamp_ms.second))
                    {
                        audio_caption = audio_captions.second[caption_count];
                        break;
                    }
                    caption_count += 1;
                }
            }
            std::cout << "Audio caption: " << audio_caption << "\n";
            std::string result = process_prompt(ctx_llava, image_embed, &params, params.prompt, video_metadata, audio_caption);
            std::cout
                << "Processed image [" << count << "/" << num_frames << "]:\n"
                << result
                << std::endl;

            count += 1;
            captions.push_back(result);
            frame_inds.push_back(frame_ind);
            // for now we are emptying the context between each image
            // TODO: sliding context window
            llama_kv_cache_seq_rm(ctx_llava->ctx_llama, 0, 0, n_ctx);
            llava_image_embed_free(image_embed);
        }

        save_data_to_file(params.doc_dir, file_name, video_path, fps, frame_inds, captions);
    }

    llava_free(ctx_llava);
    return 0;
}
