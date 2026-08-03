<?php

declare(strict_types=1);

/**
 * Minimal PSR-4 autoloader for running examples/tests without Composer
 * installed. If you `composer install`, composer's own autoloader
 * (vendor/autoload.php) supersedes this -- use that in real projects.
 */
spl_autoload_register(function (string $class): void {
    $prefix = 'FlashAttention\\';
    if (!str_starts_with($class, $prefix)) {
        return;
    }
    $relative = substr($class, strlen($prefix));
    $path = __DIR__ . '/src/' . str_replace('\\', '/', $relative) . '.php';
    if (is_file($path)) {
        require $path;
    }
});
